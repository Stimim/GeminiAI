/* Exact component enumeration and log-space conditioning for fastsolve.py. */
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <limits.h>

typedef struct {
    size_t count, capacity;
    double *hits;
    unsigned char *layouts;
    double logcount;
} Stats;
typedef struct {
    int m, nc, *cells, *need, *left, *sum, *degree, *inc;
    unsigned char *assignment;
    Stats *stats;
    size_t nodes, limit;
    int overflow, failed, remaining, probabilities_only, borrowed_stats;
    int approximate, target;
    uint64_t *rng;
    double density;
} Component;
typedef struct {
    int n, remaining, ni, nc;
    unsigned char *claimed;
    int *interior;
    Component *components;
    double *suffix;
} Solver;

/* Exact component enumerations recur across hypothetical reveals. Retain only
 * sufficient statistics, never randomized fallback results. */
typedef struct ComponentMemo {
    struct ComponentMemo *next;
    uint64_t hash;
    size_t length;
    int *key, m;
    Stats *stats;
} ComponentMemo;

static int *component_key(Component *c, size_t *length, uint64_t *hash) {
    size_t size=5+(size_t)c->nc*2+c->m;
    for (int i=0;i<c->m;++i) size+=c->degree[i];
    int *key=malloc(size*sizeof(int));
    if (!key) return NULL;
    size_t p=0;
    key[p++]=c->m; key[p++]=c->nc; key[p++]=c->remaining;
    key[p++]=(int)(uint32_t)c->limit;
    key[p++]=(int)(uint32_t)((uint64_t)c->limit>>32);
    for (int q=0;q<c->nc;++q) { key[p++]=c->need[q]; key[p++]=c->left[q]; }
    for (int i=0;i<c->m;++i) {
        key[p++]=c->degree[i];
        for (int j=0;j<c->degree[i];++j) key[p++]=c->inc[i*8+j];
    }
    *hash=1469598103934665603ULL;
    for (size_t i=0;i<size;++i) { *hash^=(uint32_t)key[i]; *hash*=1099511628211ULL; }
    *length=size;
    return key;
}

static void free_component_memo(ComponentMemo **memo) {
    for (int i=0;i<4096;++i) while (memo[i]) {
        ComponentMemo *entry=memo[i]; memo[i]=entry->next;
        for (int k=0;k<=entry->m;++k) { free(entry->stats[k].hits); free(entry->stats[k].layouts); }
        free(entry->stats); free(entry->key); free(entry);
    }
}
static uint64_t next_random(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
static double uniform(uint64_t *s) { return (next_random(s) >> 11) * 0x1.0p-53; }
static size_t bounded(uint64_t *s, size_t bound) {
    uint64_t b = (uint64_t)bound, x, threshold = -b % b;
    do { x = next_random(s); } while (x < threshold);
    return (size_t)(x % b);
}
static double addlog(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (a < b) { double t = a; a = b; b = t; }
    return a + log1p(exp(b-a));
}
static void clear_stats(Component *c) {
    if (!c->stats) return;
    for (int k=0; k<=c->m; ++k) {
        free(c->stats[k].hits); free(c->stats[k].layouts);
    }
    memset(c->stats, 0, (size_t)(c->m+1)*sizeof(Stats));
}
void fs_free(void *ptr) {
    Solver *s = ptr;
    if (!s) return;
    for (int j=0; j<s->nc; ++j) {
        Component *c = s->components+j;
        if (!c->borrowed_stats) { clear_stats(c); free(c->stats); }
        free(c->cells); free(c->need);
        free(c->left); free(c->sum); free(c->degree); free(c->inc); free(c->assignment);
    }
    free(s->components); free(s->claimed); free(s->interior); free(s->suffix); free(s);
}
static void record(Component *c, int k) {
    Stats *st = c->stats+k;
    if (!st->hits) {
        st->hits = calloc((size_t)c->m, sizeof(double));
        if (!st->hits) { c->failed=1; return; }
    }
    if (!c->probabilities_only && st->count == st->capacity) {
        size_t cap = st->capacity ? st->capacity*2 : 8;
        if (cap > SIZE_MAX/(size_t)c->m) { c->failed=1; return; }
        void *p = realloc(st->layouts, cap*(size_t)c->m);
        if (!p) { c->failed=1; return; }
        st->layouts=p; st->capacity=cap;
    }
    if (!c->probabilities_only)
        memcpy(st->layouts+st->count*(size_t)c->m, c->assignment, (size_t)c->m);
    st->count++;
    for (int i=0; i<c->m; ++i) st->hits[i] += c->assignment[i];
}
/* Always unwind mutable constraint state, including early success/overflow. */
static int dfs(Component *c, int i, int k, int randomize) {
    if (c->target>=0 && (k>c->target || c->m-i<c->target-k)) return 0;
    if (++c->nodes > c->limit) { c->overflow=1; return 0; }
    if (i == c->m) { record(c,k); return 1; }
    int first = randomize==1 && uniform(c->rng)<c->density;
    for (int branch=0; branch<2; ++branch) {
        int v=first ^ branch, ok=1;
        if (k+v>c->remaining) continue;
        for (int t=0; t<c->degree[i]; ++t) {
            int q=c->inc[i*8+t];
            c->left[q]--; c->sum[q]+=v;
            if (c->sum[q]>c->need[q] || c->sum[q]+c->left[q]<c->need[q]) ok=0;
        }
        c->assignment[i]=(unsigned char)v;
        int found = ok && dfs(c,i+1,k+v,randomize);
        for (int t=0; t<c->degree[i]; ++t) {
            int q=c->inc[i*8+t]; c->left[q]++; c->sum[q]-=v;
        }
        if (c->overflow || c->failed || (randomize && found)) return found;
    }
    return 0;
}

/* Complete missing count support only after approximate statistics fail global
 * conditioning. A deterministic, count-constrained search has no sampling
 * budget: exhaustion proves that a bucket is genuinely impossible. */
static int complete_count_support(Component *c) {
    c->limit=SIZE_MAX;
    for (int k=0;k<=c->m && k<=c->remaining;++k) if (!c->stats[k].count) {
        c->target=k; c->nodes=0; c->overflow=0;
        dfs(c,0,0,2);
        if (c->failed) { c->target=-1; return 0; }
    }
    c->target=-1;
    return 1;
}
static int root(int *parent, int x) {
    while (parent[x]!=x) { parent[x]=parent[parent[x]]; x=parent[x]; }
    return x;
}

static void *fs_create_impl(int h, int w, int mines, int remaining, const int8_t *board,
                size_t limit, uint64_t seed, double *probs, int *exact, char *error,
                int probabilities_only, ComponentMemo **memo, const double *logfactorial) {
    Solver *s=NULL;
    int *parent=NULL, *cons=NULL, *sizes=NULL, *needs=NULL, *which=NULL, *queue=NULL, *pos=NULL;
    unsigned char *adj=NULL, *seen=NULL;
    const char *message="out of memory";
    int nq=0, unknown=0, claimed=0;
    if (h<=0 || w<=0 || h>INT_MAX/w || remaining<0 || mines<0) {
        snprintf(error,256,"invalid board dimensions or mine count"); return NULL;
    }
    int n=h*w;
    if (mines>n) { snprintf(error,256,"invalid mine count"); return NULL; }
    s=calloc(1,sizeof(Solver));
    parent=malloc((size_t)n*sizeof(int)); cons=malloc((size_t)n*8*sizeof(int));
    sizes=calloc((size_t)n,sizeof(int)); needs=malloc((size_t)n*sizeof(int));
    which=malloc((size_t)n*sizeof(int)); queue=malloc((size_t)n*sizeof(int));
    pos=malloc((size_t)n*sizeof(int)); seen=calloc((size_t)n,1);
    adj=calloc((size_t)n,(size_t)n);
    if (!s || !parent || !cons || !sizes || !needs || !which || !queue || !pos || !seen || !adj) goto fail;
    s->n=n; s->remaining=remaining;
    s->claimed=calloc((size_t)n,1); s->interior=malloc((size_t)n*sizeof(int));
    s->components=calloc((size_t)n,sizeof(Component));
    if (!s->claimed || !s->interior || !s->components) goto fail;
    for (int i=0;i<n;++i) {
        parent[i]=i; which[i]=-1; pos[i]=-1; probs[i]=0;
        if (board[i]==-1) unknown++;
        else if (board[i]>=9 && board[i]<=10) { s->claimed[i]=1; claimed++; }
        else if (board[i]<0 || board[i]>8) { message="invalid board cell"; goto fail; }
    }
    if (remaining>unknown || claimed+remaining!=mines) { message="inconsistent mine count"; goto fail; }
    for (int i=0;i<n;++i) if (board[i]>=0 && board[i]<=8) {
        int r=i/w, col=i%w, nd=board[i], count=0;
        for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
            int rr=r+dr, cc=col+dc;
            if ((!dr && !dc) || rr<0 || rr>=h || cc<0 || cc>=w) continue;
            int a=rr*w+cc;
            if (board[a]==-1) cons[nq*8+count++]=a;
            else if (board[a]>=9) nd--;
        }
        if (nd<0 || nd>count) { message="inconsistent opened number"; goto fail; }
        if (!count) continue;
        sizes[nq]=count; needs[nq]=nd;
        int a=cons[nq*8];
        for (int t=0;t<count;++t) {
            int b=cons[nq*8+t]; parent[root(parent,b)]=root(parent,a);
            for (int u=0;u<count;++u) adj[(size_t)b*n+cons[nq*8+u]]=1;
        }
        nq++;
    }
    for (int i=0;i<n;++i) if (board[i]==-1) {
        if (!adj[(size_t)i*n+i]) { s->interior[s->ni++]=i; continue; }
        int r=root(parent,i);
        if (which[r]<0) which[r]=s->nc++;
        s->components[which[r]].m++;
    }
    for (int q=0;q<nq;++q) s->components[which[root(parent,cons[q*8])]].nc++;
    *exact=1;
    for (int j=0;j<s->nc;++j) {
        Component *c=s->components+j;
        c->cells=malloc((size_t)c->m*sizeof(int)); c->need=malloc((size_t)c->nc*sizeof(int));
        c->left=malloc((size_t)c->nc*sizeof(int)); c->sum=calloc((size_t)c->nc,sizeof(int));
        c->degree=calloc((size_t)c->m,sizeof(int)); c->inc=malloc((size_t)c->m*8*sizeof(int));
        c->assignment=calloc((size_t)c->m,1); c->stats=calloc((size_t)c->m+1,sizeof(Stats));
        if (!c->cells || !c->need || !c->left || !c->sum || !c->degree || !c->inc || !c->assignment || !c->stats) goto fail;
        int start=-1, best=INT_MAX;
        for (int a=0;a<n;++a) if (board[a]==-1 && adj[(size_t)a*n+a] && which[root(parent,a)]==j) {
            int degree=0, row=a/w, col=a%w;
            /* Shared-clue variables are at most two rows/columns apart. */
            for (int rr=row-2;rr<=row+2;++rr) for (int cc=col-2;cc<=col+2;++cc)
                if (rr>=0 && rr<h && cc>=0 && cc<w) degree+=adj[(size_t)a*n+rr*w+cc];
            if (degree<best) { best=degree; start=a; }
        }
        int front=0, back=0;
        queue[back++]=start; seen[start]=1;
        while (front<back) {
            int a=queue[front]; c->cells[front]=a; pos[a]=front; front++;
            int row=a/w, col=a%w;
            for (int rr=row-2;rr<=row+2;++rr) for (int cc=col-2;cc<=col+2;++cc) {
                if (rr<0 || rr>=h || cc<0 || cc>=w) continue;
                int b=rr*w+cc;
                if (adj[(size_t)a*n+b] && !seen[b]) { seen[b]=1; queue[back++]=b; }
            }
        }
        int cq=0;
        for (int q=0;q<nq;++q) if (which[root(parent,cons[q*8])]==j) {
            c->need[cq]=needs[q]; c->left[cq]=sizes[q];
            for (int t=0;t<sizes[q];++t) {
                int p=pos[cons[q*8+t]]; c->inc[p*8+c->degree[p]++]=cq;
            }
            cq++;
        }
        c->remaining=remaining; c->rng=&seed; c->density=(double)remaining/(unknown ? unknown:1); c->limit=limit;
        c->target=-1;
        c->probabilities_only=probabilities_only;
        size_t key_length=0;
        uint64_t key_hash=0;
        int *key=memo ? component_key(c,&key_length,&key_hash):NULL;
        if (memo && !key) goto fail;
        ComponentMemo *hit=memo ? memo[key_hash&4095]:NULL;
        while (hit && (hit->hash!=key_hash || hit->length!=key_length ||
                       memcmp(hit->key,key,key_length*sizeof(int)))) hit=hit->next;
        if (hit) {
            free(c->stats); c->stats=hit->stats; c->borrowed_stats=1;
            free(key);
        } else {
            dfs(c,0,0,0);
            if (c->failed) { free(key); goto fail; }
            if (c->overflow) {
                /* Retry exact search before discarding its statistics. Keep
                 * the original budget in the memo key made above. */
                clear_stats(c);
                size_t retry=limit>1000000/20 ? 1000000:limit*20;
                if (retry<limit) retry=limit;
                c->limit=retry; c->nodes=0; c->overflow=0;
                dfs(c,0,0,0);
                if (c->failed) { free(key); goto fail; }
            }
            if (c->overflow) {
                free(key); key=NULL;
                *exact=0; c->approximate=1; clear_stats(c);
                c->limit=limit>10000 ? limit:10000;
                for (int attempt=0;attempt<48;++attempt) {
                    c->nodes=0; c->overflow=0; dfs(c,0,0,1);
                    if (c->failed) goto fail;
                }
                size_t found=0;
                for (int k=0;k<=c->m;++k) found+=c->stats[k].count;
                if (!found && !complete_count_support(c)) goto fail;
            }
            if (key) {
                ComponentMemo *entry=malloc(sizeof(ComponentMemo));
                if (!entry) { free(key); goto fail; }
                *entry=(ComponentMemo){memo[key_hash&4095],key_hash,key_length,key,c->m,c->stats};
                memo[key_hash&4095]=entry; c->borrowed_stats=1;
            }
        }
        size_t count=0; for (int k=0;k<=c->m;++k) count+=c->stats[k].count;
        if (!count) { message="could not find a consistent component layout"; goto fail; }
        c->rng=NULL;
    }
    size_t stride=(size_t)remaining+1;
    s->suffix=malloc(((size_t)s->nc+1)*stride*sizeof(double));
    if (!s->suffix) goto fail;
rebuild_suffix:
    for (int j=0;j<s->nc;++j) {
        Component *c=s->components+j;
        for (int k=0;k<=c->m;++k) if (c->stats[k].count)
            c->stats[k].logcount=log((double)c->stats[k].count);
    }
    for (int r=0;r<=remaining;++r)
        s->suffix[(size_t)s->nc*stride+r]=r>s->ni ? -INFINITY :
            logfactorial ? logfactorial[s->ni]-logfactorial[r]-logfactorial[s->ni-r] :
            lgamma(s->ni+1.)-lgamma(r+1.)-lgamma(s->ni-r+1.);
    for (int j=s->nc-1;j>=0;--j) {
        Component *c=s->components+j;
        for (int r=0;r<=remaining;++r) {
            double v=-INFINITY;
            for (int k=0;k<=c->m && k<=r;++k) if (c->stats[k].count)
                v=addlog(v,c->stats[k].logcount+s->suffix[(size_t)(j+1)*stride+r-k]);
            s->suffix[(size_t)j*stride+r]=v;
        }
    }
    double z=s->suffix[remaining];
    if (!isfinite(z)) {
        int repaired=0;
        for (int j=0;j<s->nc;++j) {
            Component *c=s->components+j;
            if (!c->approximate) continue;
            if (!complete_count_support(c)) goto fail;
            c->approximate=0; repaired=1;
        }
        if (repaired) goto rebuild_suffix;
        message="no consistent global mine count"; goto fail;
    }
    double *forward=malloc(stride*sizeof(double)), *next=malloc(stride*sizeof(double));
    if (!forward || !next) { free(forward); free(next); goto fail; }
    for (int p=0;p<=remaining;++p) forward[p]=-INFINITY;
    forward[0]=0;
    for (int j=0;j<s->nc;++j) {
        Component *c=s->components+j;
        for (int p=0;p<=remaining;++p) next[p]=-INFINITY;
        for (int k=0;k<=c->m && k<=remaining;++k) if (c->stats[k].count) {
            double countlog=c->stats[k].logcount, weight=-INFINITY;
            for (int p=0;p+k<=remaining;++p) if (isfinite(forward[p])) {
                next[p+k]=addlog(next[p+k],forward[p]+countlog);
                weight=addlog(weight,forward[p]+s->suffix[(size_t)(j+1)*stride+remaining-p-k]);
            }
            double factor=exp(weight-z);
            for (int t=0;t<c->m;++t) probs[c->cells[t]]+=c->stats[k].hits[t]*factor;
        }
        double *tmp=forward; forward=next; next=tmp;
    }
    double interior_p=0;
    if (s->ni) for (int p=0;p<=remaining;++p)
        interior_p+=exp(forward[p]+s->suffix[(size_t)s->nc*stride+remaining-p]-z)*(remaining-p)/(double)s->ni;
    for (int i=0;i<s->ni;++i) probs[s->interior[i]]=interior_p;
    free(forward); free(next);
    free(parent); free(cons); free(sizes); free(needs); free(which); free(queue); free(pos); free(seen); free(adj);
    return s;
fail:
    snprintf(error,256,"%s",message);
    free(parent); free(cons); free(sizes); free(needs); free(which); free(queue); free(pos); free(seen); free(adj);
    fs_free(s); return NULL;
}

void *fs_create(int h, int w, int mines, int remaining, const int8_t *board,
                size_t limit, uint64_t seed, double *probs, int *exact, char *error) {
    return fs_create_impl(h,w,mines,remaining,board,limit,seed,probs,exact,error,0,NULL,NULL);
}

int fs_batch_probs(int h, int w, int mines, const int8_t *boards,
                   const int32_t *scores, size_t count, size_t limit,
                   uint64_t seed, double *probs, size_t *failed, char *error) {
    ComponentMemo *memo[4096]={0};
    double *logfactorial=NULL;
    int ok=0;
    *failed=0;
    if (h<=0 || w<=0 || h>INT_MAX/w || mines<0 || mines>h*w) {
        snprintf(error,256,"invalid board dimensions or mine count"); goto done;
    }
    int n=h*w;
    if (!count) return 1;
    logfactorial=malloc(((size_t)n+1)*sizeof(double));
    if (!logfactorial) { snprintf(error,256,"out of memory"); goto done; }
    for (int i=0;i<=n;++i) logfactorial[i]=lgamma(i+1.);
    for (size_t i=0;i<count;++i) {
        *failed=i;
        int64_t remaining=(int64_t)mines-scores[2*i]-scores[2*i+1];
        if (remaining<0 || remaining>n) {
            snprintf(error,256,"invalid mine count"); goto done;
        }
        int exact;
        void *solver=fs_create_impl(h,w,mines,(int)remaining,boards+i*n,
                                   limit,next_random(&seed),probs+i*n,&exact,
                                   error,1,memo,logfactorial);
        if (!solver) goto done;
        fs_free(solver);
    }
    ok=1;
done:
    free(logfactorial);
    free_component_memo(memo);
    return ok;
}

int fs_sample_many(void *ptr, size_t count, uint64_t seed, unsigned char *out) {
    Solver *s=ptr;
    size_t stride=(size_t)s->remaining+1;
    int *pool=malloc((size_t)(s->ni ? s->ni:1)*sizeof(int));
    if (!pool) return 0;
    for (size_t row=0;row<count;++row) {
        unsigned char *layout=out+row*(size_t)s->n;
        memcpy(layout,s->claimed,(size_t)s->n);
        int remaining=s->remaining;
        for (int j=0;j<s->nc;++j) {
            Component *c=s->components+j;
            double target=uniform(&seed), cumulative=0;
            int chosen=-1;
            double denominator=s->suffix[(size_t)j*stride+remaining];
            for (int k=0;k<=c->m && k<=remaining;++k) if (c->stats[k].count) {
                double weight=exp(log((double)c->stats[k].count)+s->suffix[(size_t)(j+1)*stride+remaining-k]-denominator);
                if (weight==0) continue;
                chosen=k; cumulative+=weight;
                if (target<cumulative) break;
            }
            if (chosen<0) { free(pool); return 0; }
            Stats *st=c->stats+chosen;
            const unsigned char *assignment=st->layouts+bounded(&seed,st->count)*(size_t)c->m;
            for (int t=0;t<c->m;++t) layout[c->cells[t]]=assignment[t];
            remaining-=chosen;
        }
        memcpy(pool,s->interior,(size_t)s->ni*sizeof(int));
        for (int t=0;t<remaining;++t) {
            int q=t+(int)bounded(&seed,(size_t)(s->ni-t));
            int tmp=pool[t]; pool[t]=pool[q]; pool[q]=tmp; layout[pool[t]]=1;
        }
    }
    free(pool); return 1;
}

/* Decision-local memoization also preserves Python's first-seen outcome order.
 * Seeds are supplied by NumPy, one per cache miss, including exact analyses. */
typedef struct {
    uint64_t hash;
    int8_t *board;
    int score0, score1;
    double advantage;
} ExpertEntry;
typedef struct {
    int h, w, n, mines;
    size_t capacity, used;
    ExpertEntry *entries;
    const uint64_t *seeds;
    double *probs, alpha, cap;
    char *error;
    ComponentMemo **memo;
    double *logfactorial;
} ExpertCache;

static uint64_t expert_hash(const int8_t *board, int n, int a, int b) {
    uint64_t hash=1469598103934665603ULL;
    for (int i=0;i<n;++i) { hash^=(unsigned char)board[i]; hash*=1099511628211ULL; }
    hash^=(uint64_t)a; hash*=1099511628211ULL; hash^=(uint64_t)b;
    return hash;
}

static double expert_guess(double p, double cap) {
    if (p>cap) p=cap;
    return p/(1-p);
}

static int expert_advantage(ExpertCache *cache, const int8_t *board,
                            int a, int b, double *value) {
    uint64_t hash=expert_hash(board,cache->n,a,b);
    size_t slot=hash&(cache->capacity-1);
    while (cache->entries[slot].board) {
        ExpertEntry *entry=cache->entries+slot;
        if (entry->hash==hash && entry->score0==a && entry->score1==b &&
            !memcmp(entry->board,board,(size_t)cache->n)) {
            *value=entry->advantage; return 1;
        }
        slot=(slot+1)&(cache->capacity-1);
    }
    int exact;
    void *solver=fs_create_impl(cache->h,cache->w,cache->mines,cache->mines-a-b,
                           board,50000,cache->seeds[cache->used],cache->probs,
                           &exact,cache->error,1,cache->memo,cache->logfactorial);
    if (!solver) return 0;
    fs_free(solver);
    int sure=0;
    double best=0;
    for (int i=0;i<cache->n;++i) if (board[i]==-1) {
        if (cache->probs[i]>=1-1e-9) sure++;
        else if (cache->probs[i]>best) best=cache->probs[i];
    }
    /* An empty remainder contributes zero, even for an unusual negative cap. */
    int rest=0;
    for (int i=0;i<cache->n;++i)
        if (board[i]==-1 && cache->probs[i]<1-1e-9) { rest=1; break; }
    *value=sure+(rest ? cache->alpha*expert_guess(best,cache->cap):0);
    int8_t *copy=malloc((size_t)cache->n);
    if (!copy) { snprintf(cache->error,256,"out of memory"); return 0; }
    memcpy(copy,board,(size_t)cache->n);
    cache->entries[slot]=(ExpertEntry){hash,copy,a,b,*value};
    cache->used++;
    return 1;
}

static void expert_reveal(int8_t *board, const unsigned char *counts,
                          const int *cells, int length, const int *nb,
                          const int *degree, int *stack) {
    int top=0;
    for (int i=0;i<length;++i) {
        int c=cells[i];
        if (board[c]==-1) {
            board[c]=(int8_t)counts[c];
            if (!counts[c]) stack[top++]=c;
        }
    }
    while (top) {
        int c=stack[--top];
        for (int j=0;j<degree[c];++j) {
            int x=nb[c*8+j];
            if (board[x]==-1) {
                board[x]=(int8_t)counts[x];
                if (!counts[x]) stack[top++]=x;
            }
        }
    }
}

/* Bomb centres and their expected gains use NumPy's ordering in the caller. */
int fs_expert_scores(int h, int w, int mines, const int8_t *board,
                     const int *scores, int me, const unsigned char *layouts,
                     size_t samples, const double *probs, const int *candidates,
                     size_t nc, const int *bombs, const double *expected, size_t nbombs,
                     double alpha, double cap, double reserve, const uint64_t *seeds,
                     int *actions, double *values, size_t *nout, size_t *used,
                     char *error) {
    int n=h*w, ok=0;
    size_t maximum=(nc+nbombs)*samples, capacity=8;
    while (capacity/2<maximum) {
        if (capacity>SIZE_MAX/2) { snprintf(error,256,"workspace too large"); return 0; }
        capacity*=2;
    }
    ComponentMemo *memo[4096]={0};
    ExpertCache cache={h,w,n,mines,capacity,0,NULL,seeds,NULL,alpha,cap,error,memo,NULL};
    cache.entries=calloc(capacity,sizeof(ExpertEntry));
    cache.probs=malloc((size_t)n*sizeof(double));
    cache.logfactorial=malloc(((size_t)n+1)*sizeof(double));
    int *nb=malloc((size_t)n*8*sizeof(int)), *degree=calloc((size_t)n,sizeof(int));
    int *stack=malloc((size_t)n*sizeof(int));
    unsigned char *counts=calloc(samples,(size_t)n);
    int8_t *outcomes=malloc((samples ? samples:1)*(size_t)n);
    size_t *weights=malloc((samples ? samples:1)*sizeof(size_t));
    int8_t *current=malloc((size_t)n);
    if (!cache.entries || !cache.probs || !cache.logfactorial || !nb || !degree || !stack ||
        !counts || !outcomes || !weights || !current) {
        snprintf(error,256,"out of memory"); goto done;
    }
    for (int i=0;i<=n;++i) cache.logfactorial[i]=lgamma(i+1.);
    for (int c=0;c<n;++c) {
        int r=c/w, col=c%w;
        for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
            int rr=r+dr, cc=col+dc;
            if ((!dr && !dc) || rr<0 || rr>=h || cc<0 || cc>=w) continue;
            nb[c*8+degree[c]++]=rr*w+cc;
        }
        for (size_t s=0;s<samples;++s)
            for (int j=0;j<degree[c];++j)
                counts[s*n+c]+=layouts[s*n+nb[c*8+j]];
    }
    *nout=0;
    for (size_t i=0;i<nc;++i) {
        int c=candidates[i];
        size_t distinct=0, safe=0;
        for (size_t s=0;s<samples;++s) if (!layouts[s*n+c]) {
            memcpy(current,board,(size_t)n);
            expert_reveal(current,counts+s*n,&c,1,nb,degree,stack);
            size_t j=0;
            for (;j<distinct;++j)
                if (!memcmp(outcomes+j*n,current,(size_t)n)) break;
            if (j==distinct) {
                memcpy(outcomes+j*n,current,(size_t)n); weights[j]=0; distinct++;
            }
            weights[j]++; safe++;
        }
        if (!safe) continue;
        double advantage=0;
        for (size_t j=0;j<distinct;++j) {
            double value;
            if (!expert_advantage(&cache,outcomes+j*n,scores[0],scores[1],&value)) goto done;
            advantage+=weights[j]*value;
        }
        actions[*nout]=c;
        values[(*nout)++]=expert_guess(probs[c],cap)-advantage/safe;
    }
    for (size_t i=0;i<nbombs;++i) {
        int centre=bombs[i], cells[25], length=0;
        int r=centre/w, col=centre%w;
        for (int rr=r-2;rr<=r+2;++rr) for (int cc=col-2;cc<=col+2;++cc)
            if (rr>=0 && rr<h && cc>=0 && cc<w) cells[length++]=rr*w+cc;
        double advantage=0;
        for (size_t s=0;s<samples;++s) {
            int sc[2]={scores[0],scores[1]};
            memcpy(current,board,(size_t)n);
            for (int j=0;j<length;++j) {
                int c=cells[j];
                if (current[c]==-1 && layouts[s*n+c]) { current[c]=9+me; sc[me]++; }
            }
            expert_reveal(current,counts+s*n,cells,length,nb,degree,stack);
            double value;
            if (!expert_advantage(&cache,current,sc[0],sc[1],&value)) goto done;
            advantage+=value;
        }
        actions[*nout]=n+centre;
        values[(*nout)++]=expected[i]-advantage/samples-reserve;
    }
    ok=1;
done:
    *used=cache.used;
    if (cache.entries) for (size_t i=0;i<capacity;++i) free(cache.entries[i].board);
    free(cache.entries); free(cache.probs); free(nb); free(degree); free(stack);
    free(counts); free(outcomes); free(weights); free(current);
    free(cache.logfactorial);
    free_component_memo(memo);
    return ok;
}
