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

import com.google.gwt.user.client.rpc.IsSerializable;

/**
 * @author stimim
 * 
 * if a square is an instance of MineSquare, there must be a mine.
 */
public class MineSquare extends KnownSquare implements IsSerializable {
	public static MineSquare FoundMine = null ;
	/**
	 * -1 : player
	 *  0 : nobody
	 *  1 : AI (me)
	 */
	public int owner ;
	public MineSquare(int x, int y, int owner) {
		super(x, y);
		// TODO Auto-generated constructor stub
		this.owner = owner ;
		
		if( this.owner == 0 ) {
			MineSquare.FoundMine = this ;
		}
	}
}
