/*   This file is part of GeminiAI.
 *   GeminiAI is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   GeminiAI is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Gemini.  If not, see <http://www.gnu.org/licenses/>.
 *
 *   Author: Stimim
 *   Email : death1048576@gmail.com
 */
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
