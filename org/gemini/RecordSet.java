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
 *   along with GeminiAI.  If not, see <http://www.gnu.org/licenses/>.
 *
 *   Author: Stimim
 *   Email : death1048576@gmail.com
 */
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
