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
import java.util.Random;
import java.util.Set;

import org.gemini.Square;
import org.gemini.UnknownSquare;
import org.psmonkey.product.client.mine.vo.GameInfo;
import org.psmonkey.product.server.mine.AI_Interface;

@SuppressWarnings("serial")
public class GeminiAI implements AI_Interface {
	/**
	 * Variables
	 */
	public static String NAME = "Gemini";
	Random random = new Random();
	GameInfo gameInfo;
	int totalSet;
	Square[][] map;
	Set<Section> sections ;

	public GeminiAI() {

	}

	private void initial(GameInfo info) {
		gameInfo = info;

		map = new Square[info.getWidth()][info.getHeight()];

		NumberedSquare.WIDTH = info.getWidth();
		NumberedSquare.HEIGHT = info.getHeight();

		for (int i = 0; i < info.getWidth(); i++) {
			for (int j = 0; j < info.getHeight(); j++) {
				int value = info.getMap()[i][j];
				switch (value) {
				// default:
				case -1: // UNKNOWN
					map[i][j] = new UnknownSquare(i, j);
					break;
				case 1: case 2: case 3: case 4:
				case 5: case 6: case 7: case 8:
					map[i][j] = new NumberedSquare(i, j, value);
					break;
				case 0:
					map[i][j] = new ZeroSquare(i, j);
					break;
				case 9: case -9:
					map[i][j] = new MineSquare(i, j, value / 9);
					break;
				}
			}
		}
		totalSet = 0;
		sections = new HashSet<Section>();
	}

	private void prepare() {
		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof NumberedSquare)
					((NumberedSquare) map[i][j]).checkNeighbor(map);
			}
		}

		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof NumberedSquare)
					((NumberedSquare) map[i][j]).checkTrivial(map);
			}
		}
	}

	/**
	 * Separate the map into some sections
	 */
	private void separateMap() {
		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof UnknownSquare) {
					((UnknownSquare) map[i][j]).unionNeighbors();
				}
			}
		}

		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof NumberedSquare) {
					NumberedSquare v = (NumberedSquare) map[i][j];
					if (v.belongsTo != null) {
						if (!sections.contains(v.belongsTo)) {
							sections.add(v.belongsTo);
						}
					}
				}
			}
		}
	}

	private void destroy() {
		for (int i = 0; i < gameInfo.getWidth(); i++)
			for (int j = 0; j < gameInfo.getHeight(); j++)
				map[i][j] = null;
		map = null;

		if (sections != null) {
			for ( Section section : sections) {
				for (NumberedSquare v : section) {
					if (v.neighbors != null) {
						v.neighbors.clear();
						v.neighbors = null;
					}
				}
			}
			sections.clear();
		}
		sections = null;
	}

	public double isBlank(Square square) {
		double pBlank = 1;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				if (dx != 0 || dy != 0) {
					int x_ = square.x + dx;
					int y_ = square.y + dy;
					if (x_ >= 0 && x_ < gameInfo.getWidth() && y_ >= 0
							&& y_ < gameInfo.getHeight()) {
						if (map[x_][y_] instanceof UnknownSquare) {
							double chance = ((UnknownSquare) map[x_][y_]).isMineChance;
							pBlank *= (1 - chance);
						}
						if (map[x_][y_] instanceof MineSquare) {
							pBlank = 0;
						}
					}
				}
			}
		}
		return pBlank;
	}

	public int scoreUnknown(UnknownSquare square) {
		//double isMineChance = square.isMineChance;
		//if( square.expectValue > 0.00 )
		//	System.out.printf( "(% 2d,% 2d) = %.2f\n", square.x , square.y , square.expectValue ) ;
		double v = square.isMineChance , u = square.expectValue , rv ;
		if (v >= 0.99)
			rv = 10000 + (v * 10000);
		else if (v >= 0.799999)
			rv = 4000 +  (v * 10000);
		else if (v >= 0.499999)
			rv = (v * 10000 * (1 - isBlank(square))) ;
		else
			rv = (v * 9000 * (1 - isBlank(square)));
		
		return (int) (rv * ( 1 + u * 10 ) * 100) ;
	}

	public int scoreZero(Square square) {
		double score = 0;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dy = -1; dy <= 1; dy++) {
				if (dx != 0 || dy != 0) {
					int x_ = square.x + dx;
					int y_ = square.y + dy;
					if (x_ >= 0 && x_ < gameInfo.getWidth() && y_ >= 0
							&& y_ < gameInfo.getHeight()) {
						if (map[x_][y_] instanceof KnownSquare)
							score += 500;
						if (map[x_][y_] instanceof UnknownSquare) {
							double chance = ((UnknownSquare) map[x_][y_]).isMineChance;
							if (chance <= 0.05 || chance > 0.99)
								score += 700;
						}
						if (map[x_][y_] instanceof MineSquare) {
							score += 900;
						}
					} else {
						score += 600;
					}
				}
			}
		}
		return (int) (score * (1 - isBlank(square)));
	}
	
	public Square computeScore() {
		Square result = null;
		int n = 1;
		int score = -10000;
		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof UnknownSquare) {
					int newScoreU = scoreUnknown((UnknownSquare) map[i][j]);
					int newScoreZ = scoreZero(map[i][j]);
					int newScore = newScoreU > newScoreZ ? newScoreU
							: newScoreZ;
					if ( newScore == score ) {
						if (random.nextDouble() > (double) n / (n + 1)) {
							score = newScore;
							result = map[i][j];
						}
						n++;
					}
					if (newScore > score) {
						score = newScore;
						result = map[i][j];
						n = 1;
					}
				}
				if (map[i][j] instanceof ZeroSquare)
					if (((ZeroSquare) map[i][j]).iFound) {
						int newScore = scoreZero(map[i][j]);
						if (newScore == score) {
							if (random.nextDouble() > (double) n / (n + 1)) {
								result = map[i][j];
							}
							n++;
						}

						if (newScore > score) {
							score = newScore;
							result = map[i][j];
							n = 1;
						}
					}
			}
		}

		// System.out.printf( "(%d,%d) %f\n" , result.x , result.y , score ) ;
		return result;
	}

	public void computeExpectValue() {
		double expectValue = 0.0;
		if (sections != null && !sections.isEmpty()) {
			for (Section section : sections) {
				computeProb(section);
				Set<UnknownSquare> set = new HashSet<UnknownSquare>();
				for (NumberedSquare v : section) {
					set.addAll(v.neighbors);
				}

				for (UnknownSquare v : set) {
					expectValue += v.isMineChance;
				}
			}
		}
		int total = 0;
		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof UnknownSquare) {
					UnknownSquare v = (UnknownSquare) map[i][j];
					if (v.neighbors == null || v.neighbors.isEmpty()) {
						v.isMineChance = gameInfo.getRemainder() - expectValue;
						total++;
					}
				}
			}
		}
		for (int i = 0; i < gameInfo.getWidth(); i++) {
			for (int j = 0; j < gameInfo.getHeight(); j++) {
				if (map[i][j] instanceof UnknownSquare) {
					UnknownSquare v = (UnknownSquare) map[i][j];
					if (v.neighbors == null || v.neighbors.isEmpty()) {
						v.isMineChance /= total;
					}
				}
			}
		}
	}
	@Override
	public void guess(GameInfo info, int[] xy) {
		initial(info);
		prepare();
		Square v = null ;
		if (MineSquare.FoundMine != null) {
			xy[0] = MineSquare.FoundMine.x;
			xy[1] = MineSquare.FoundMine.y;
			MineSquare.FoundMine = null;
		} else {
			separateMap();
			computeExpectValue();
			if ((v = computeScore()) != null) {
				xy[0] = v.x;
				xy[1] = v.y;
			} else {
				do {
					xy[0] = random.nextInt(info.getWidth());
					xy[1] = random.nextInt(info.getHeight());
				} while (info.getMap()[xy[0]][xy[1]] != -1);
			}
		}
		destroy();
	}
}
