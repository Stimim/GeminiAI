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

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import com.google.gwt.user.client.rpc.IsSerializable;

public class UnknownSquare extends Square implements IsSerializable {
	/**
	 * 0 : is not assumed
	 * 1 : assume as mine 
	 * 2 : assume as zero
	 */
	public int assumed;
	public double expectValue ;
	public int isMineCount;
	public double isMineChance;
	/**
	 * 這一格有哪些鄰居是 NumberedSquare
	 */
	public Set<NumberedSquare> neighbors;

	public UnknownSquare(int x, int y) {
		super(x, y);
		this.isMineCount = 0;
		this.isMineChance = 0.0;
		this.assumed = 0;
		this.neighbors = null ;
	}

	public void addNeighbor(NumberedSquare square) {
		if( this.neighbors == null )
			this.neighbors = new HashSet<NumberedSquare>() ;
		this.neighbors.add(square);
	}

	public void unionNeighbors() {
		if (this.neighbors == null || this.neighbors.isEmpty())
			return;
		Section result = null;
		/**
		 * Find the largest section
		 */
		for (Iterator<NumberedSquare> it = this.neighbors.iterator(); it
				.hasNext();) {
			NumberedSquare v = it.next();
			if (v.belongsTo == null)
				continue; // skip
			if (result == null || result.size() < v.belongsTo.size())
				result = v.belongsTo;
		}
		/**
		 * If all of neighbors don't belong to any sections, create a new
		 * section
		 */
		if (result == null) {
			result = new Section();
		}
		/**
		 * Merge each section
		 */
		for (Iterator<NumberedSquare> it = this.neighbors.iterator(); it
				.hasNext();) {
			NumberedSquare v = it.next();
			if (v.belongsTo == null) {
				result.add(v);
				v.belongsTo = result;
			} else if (v.belongsTo != result) {
				result.addAll(v.belongsTo);
				for (Iterator<NumberedSquare> it2 = v.belongsTo.iterator(); it2
						.hasNext();) {
					it2.next().belongsTo = result;
				}
			}
		}
	}
}
