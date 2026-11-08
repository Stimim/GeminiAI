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
} Stats;
typedef struct {
    int m, nc, *cells, *need, *left, *sum, *degree, *inc;
    unsigned char *assignment;
    Stats *stats;
    size_t nodes, limit;
    int overflow, failed, remaining;
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
        clear_stats(c); free(c->stats); free(c->cells); free(c->need);
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
    if (st->count == st->capacity) {
        size_t cap = st->capacity ? st->capacity*2 : 8;
        if (cap > SIZE_MAX/(size_t)c->m) { c->failed=1; return; }
        void *p = realloc(st->layouts, cap*(size_t)c->m);
        if (!p) { c->failed=1; return; }
        st->layouts=p; st->capacity=cap;
    }
    memcpy(st->layouts+st->count*(size_t)c->m, c->assignment, (size_t)c->m);
    st->count++;
    for (int i=0; i<c->m; ++i) st->hits[i] += c->assignment[i];
}
/* Always unwind mutable constraint state, including early success/overflow. */
static int dfs(Component *c, int i, int k, int randomize) {
    if (++c->nodes > c->limit) { c->overflow=1; return 0; }
    if (i == c->m) { record(c,k); return 1; }
    int first = randomize && uniform(c->rng)<c->density;
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
static int root(int *parent, int x) {
    while (parent[x]!=x) { parent[x]=parent[parent[x]]; x=parent[x]; }
    return x;
}

void *fs_create(int h, int w, int mines, int remaining, const int8_t *board,
                size_t limit, uint64_t seed, double *probs, int *exact, char *error) {
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
            int degree=0; for (int b=0;b<n;++b) degree+=adj[(size_t)a*n+b];
            if (degree<best) { best=degree; start=a; }
        }
        int front=0, back=0;
        queue[back++]=start; seen[start]=1;
        while (front<back) {
            int a=queue[front]; c->cells[front]=a; pos[a]=front; front++;
            for (int b=0;b<n;++b) if (adj[(size_t)a*n+b] && !seen[b]) { seen[b]=1; queue[back++]=b; }
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
        dfs(c,0,0,0);
        if (c->failed) goto fail;
        if (c->overflow) {
            *exact=0; clear_stats(c);
            c->limit=limit>10000 ? limit:10000;
            for (int attempt=0;attempt<48;++attempt) {
                c->nodes=0; c->overflow=0; dfs(c,0,0,1);
                if (c->failed) goto fail;
            }
        }
        size_t count=0; for (int k=0;k<=c->m;++k) count+=c->stats[k].count;
        if (!count) { message="could not find a consistent component layout"; goto fail; }
        c->rng=NULL;
    }
    size_t stride=(size_t)remaining+1;
    s->suffix=malloc(((size_t)s->nc+1)*stride*sizeof(double));
    if (!s->suffix) goto fail;
    for (int r=0;r<=remaining;++r) s->suffix[(size_t)s->nc*stride+r]=r<=s->ni ? lgamma(s->ni+1.)-lgamma(r+1.)-lgamma(s->ni-r+1.) : -INFINITY;
    for (int j=s->nc-1;j>=0;--j) {
        Component *c=s->components+j;
        for (int r=0;r<=remaining;++r) {
            double v=-INFINITY;
            for (int k=0;k<=c->m && k<=r;++k) if (c->stats[k].count)
                v=addlog(v,log((double)c->stats[k].count)+s->suffix[(size_t)(j+1)*stride+r-k]);
            s->suffix[(size_t)j*stride+r]=v;
        }
    }
    double z=s->suffix[remaining];
    if (!isfinite(z)) { message="no consistent global mine count"; goto fail; }
    double *forward=malloc(stride*sizeof(double)), *next=malloc(stride*sizeof(double));
    if (!forward || !next) { free(forward); free(next); goto fail; }
    for (int p=0;p<=remaining;++p) forward[p]=-INFINITY;
    forward[0]=0;
    for (int j=0;j<s->nc;++j) {
        Component *c=s->components+j;
        for (int p=0;p<=remaining;++p) next[p]=-INFINITY;
        for (int k=0;k<=c->m && k<=remaining;++k) if (c->stats[k].count) {
            double countlog=log((double)c->stats[k].count), weight=-INFINITY;
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
