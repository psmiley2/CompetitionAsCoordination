import globalVars as globs

import evolution as ev

import board as brd
import hyperParameters as hp

dst = "data/r1.json"

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    evolution = ev.Evolution(board, dst)

    evolution.run()


if __name__ == "__main__":
    main()
