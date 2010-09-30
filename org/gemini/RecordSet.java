package org.gemini;

import java.util.HashSet;
import java.util.Set;

public class RecordSet {
	public Set<AssumptionRecord> records ;
	
	public RecordSet() {
		this.records = null ;
	}

	public void add( AssumptionRecord v ) {
		if( this.records == null )
			this.records = new HashSet<AssumptionRecord>() ;
		this.records.add(v) ;
	}
	
	public RecordSet onCondiction( int x , int y , boolean isMine ) {
		
		RecordSet result = new RecordSet() ;
		if( this.records != null ) {
			for( AssumptionRecord r : this.records ) {
				if( r.contains(x , y , isMine) )
					result.add(r) ;
			}
		}
		
		return result ;
	}
}
