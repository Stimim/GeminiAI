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

import com.google.gwt.user.client.rpc.IsSerializable;

public class ZeroSquare extends KnownSquare implements IsSerializable {
	/**
	 * false: 原地圖上是 Zero 
	 * true : 原地圖上是 Unknown
	 */
	public boolean iFound;

	public ZeroSquare(int x, int y) {
		super(x, y);
		this.iFound = false ;
	}

	public ZeroSquare(int x, int y, boolean flag) {
		super(x, y);
		// TODO Auto-generated constructor stub
		this.iFound = flag;
	}
}
