/**
 *   This file is part of GeminiAI.
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
 *   @author stimim
 *   email:  death1048576@gmail.com
 */
package org.gemini;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class Section extends HashSet<NumberedSquare> {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public Map<Integer,RecordSet> mapping ;
	
	public Section() {
		super() ;
		mapping = new HashMap<Integer,RecordSet>() ;
	}
	
	public void add( AssumptionRecord record ) {
		Integer num = record.totalMine ;
		RecordSet recordSet = null ;
		if( !mapping.containsKey(num) ) {
			recordSet = new RecordSet() ;
			mapping.put(num, recordSet) ;
		}
		else
			recordSet = mapping.get(num) ;
		recordSet.add(record) ;
	}
	
	public void makeAssumption() {
		NumberedSquare[] array = new NumberedSquare[this.size()];
		int i = 0 ;
		
		for (NumberedSquare v : this) {
			array[i] = v;
			i++;
		}
		
		Set<UnknownSquare> set = new HashSet<UnknownSquare>();
		
		for (i = 0; i < array.length; i++)
			set.addAll(array[i].neighbors);

		for (i = 0; 0 <= i && i <= array.length;) {
			if (i == array.length) {
				i--;
				AssumptionRecord record = new AssumptionRecord() ;
				for (UnknownSquare v : set) {
					switch (v.assumed) {
					default:
						break;
					case 1:
					case 2:
						record.add(v.x , v.y , (v.assumed==1) ) ;
						break;
					}
				}
				this.add(record) ;
			} else {
				i += (array[i].makeAssumption() ? 1 : -1);
				// true => made an assumption successfully
				// false=> failed to make an assumption
			}
		}
		return ;
	}
// TODO Fix the following code
//	public void tmp() {
//		for (UnknownSquare v : set) {
//			double p = v.isMineChance = (double) v.isMineCount / totalCount;
//			
//			RecordSet[] subSet = new RecordSet[2] ;
//			subSet[0] = section.recordSet.onCondiction(v.x , v.y , false) ;
//			subSet[1] = section.recordSet.onCondiction(v.x , v.y , true ) ;
//			
//			int[] eValue = { 0, 0 };
//			int[] total = { 0, 0 };
//			int[] max = {0,0} ;
//			
//			Map<String, Integer>[] mapping = (Map<String, Integer>[]) (new Map[2]);
//
//			mapping[0] = new HashMap<String, Integer>();
//			mapping[1] = new HashMap<String, Integer>() ;
//			
//			for( i = 0 ; i < 2 ; i ++ ) {
//				if (subSet[i].records != null) {
//					for (AssumptionRecord record : subSet[i].records) {
//						if (record.squares != null) {
//							total[i]++;
//							for (String str : record.squares) {
//								if (str.endsWith(":1")) {
//									String t = str.substring( 0 , str.lastIndexOf(':') ) ;
//									int count = 1;
//									if (mapping[i].containsKey(t)) {
//										count += mapping[i].get(t);
//									}
//									mapping[i].put(t, count);
//								}
//							}
//						}
//					}
//				}
//			}
//			
//			for( i = 0 ; i < 2 ; i ++ ) {
//				for( String key : mapping[i].keySet() ) {
//					int value = mapping[i].get(key) ;
//					if( value == total[i] ) {
//						eValue[i] += total[i] ;
//					}
//					else if( max[i] < value )
//						max[i] = value ;
//				}
//				eValue[i] += max[i] ;
//			}
//			//System.out.printf("(%2d,%2d) : %.2f   %d / %d   %d / %d\n", v.x , v.y , p , eValue[1] , total[1] , eValue[0] , total[0]) ;
//			v.expectValue = 0.0 ;
//			if( total[1] != 0 )
//				v.expectValue = p * (double) eValue[1] / total[1];
//			if( total[0] != 0 )
//				v.expectValue -= (1 - p) * (double) eValue[0] / total[0];
//			
//			for( i = 0 ; i < 2 ; i ++ )
//				mapping[i].clear() ;
//		}
//		
//		return section ;
//	}
}