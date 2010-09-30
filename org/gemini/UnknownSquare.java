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
