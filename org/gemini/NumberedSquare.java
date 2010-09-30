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

import com.google.gwt.user.client.rpc.IsSerializable;

/**
 * if a square is a NumberedSquare, it is opened and not zero.
 * 
 * @author stimim
 * 
 */
public class NumberedSquare extends KnownSquare implements IsSerializable {
	static public int WIDTH;
	static public int HEIGHT;
	/**
	 * 這一格屬於哪一個 Set
	 */
	public Section belongsTo;
	/**
	 * 這一格的數字是多少
	 */
	public int number;
	/**
	 * 鄰居中還有多少地雷未被找到
	 */
	public int leftMine;
	/**
	 * 在地圖上和他相鄰的未知格有多少格
	 */
	public int totalNeighbors;
	/**
	 * 型態為 UnknownSquare 的鄰居的集合
	 */
	public Set<UnknownSquare> neighbors;
	/**
	 * 猜測
	 */
	private Assumption assumption;

	public boolean makeAssumption() {
		if(this.neighbors == null ) {
			return true ;
		}
		
		boolean flag;
		if (assumption == null) {
			assumption = new Assumption(leftMine);
			flag = assumption.initialize(this.neighbors);
		} else {
			flag = assumption.nextPermutation();
		}

		if (flag) {
			assumption.applyAssumption();
			return true;
		} else {
			assumption.restore();
			assumption = null;
			return false;
		}
	}

	public NumberedSquare(int x, int y, int number) {
		super(x, y);
		this.leftMine = this.number = number;
		this.totalNeighbors = 0;
		this.neighbors = null ; //new HashSet<UnknownSquare>();
		this.belongsTo = null;
		this.assumption = null;
	}

	public void checkNeighbor(Square[][] map) {
		// Clear out previous result
		if (this.neighbors == null)
			this.neighbors = new HashSet<UnknownSquare>();
		this.neighbors.clear();
		this.leftMine = this.number;
		this.totalNeighbors = 0;

		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				if (this.x + dx >= WIDTH || this.x + dx < 0)
					continue;
				if (this.y + dy >= HEIGHT || this.y + dy < 0)
					continue;

				Square square = map[this.x + dx][this.y + dy];

				if (square instanceof UnknownSquare) {
					this.totalNeighbors++;
					this.neighbors.add((UnknownSquare) square);
					((UnknownSquare) square).addNeighbor(this);
				} else if (square instanceof MineSquare) {
					leftMine--;
				}
			}
		}
	}

	public void checkTrivial(Square[][] map) {
		if (this.neighbors == null || this.neighbors.isEmpty())
			return;
		Set<NumberedSquare> updateList = null;

		if (leftMine == 0 || leftMine == totalNeighbors) {
			updateList = new HashSet<NumberedSquare>();
			/**
			 * 把 this.neighbors 中的格子的 neighbors 加入 updateList 等一下才能更新
			 */
			for (UnknownSquare v : this.neighbors) {
				updateList.addAll(v.neighbors);
			}
		}

		if (leftMine == 0) {
			/**
			 * 把剩下的 Unknown 變成 Zero
			 */
			for (UnknownSquare v : this.neighbors) {
				int x = v.x;
				int y = v.y;
				map[x][y] = new ZeroSquare(x, y, true);
			}
		}

		if (leftMine == totalNeighbors) {
			/**
			 * 把剩下的 Unknown 變成 Mine
			 */
			for (UnknownSquare v : this.neighbors) {
				int x = v.x;
				int y = v.y;
				map[x][y] = new MineSquare(x, y, 0);
			}
		}

		if (leftMine == 0 || leftMine == totalNeighbors) {
			this.neighbors.clear();
			this.neighbors = null;
		}

		if (updateList != null && !updateList.isEmpty()) {
			if (updateList.contains(this))
				updateList.remove(this);

			for (NumberedSquare v : updateList) {
				v.checkNeighbor(map);
			}
			for (NumberedSquare v : updateList) {
				v.checkTrivial(map);
			}
		}
	}
}
