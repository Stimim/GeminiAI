package org.gemini;

import java.util.HashSet;
import java.util.Set;

public class AssumptionRecord {
	// Each square : "x:y:0" => (x,y) is not mine
	//               "x:y:1" => (x,y) is mine
	public Set<String> squares ;
	
	public AssumptionRecord() {
		this.squares = null ;
	}
	
	public void add( String v ) {
		if( this.squares == null )
			this.squares = new HashSet<String>() ;
		this.squares.add(v) ;
	}
	
	public boolean contains( int x , int y , boolean isMine ) {
		if( this.squares == null )
			return false ;
		if( isMine )
			return this.squares.contains(x+":"+y+":1") ;
		else
			return this.squares.contains(x+":"+y+":0") ;
	}
	
	public boolean contains( String v ) {
		if( this.squares == null )
			return false ;
		return this.squares.contains(v) ;
	}
}