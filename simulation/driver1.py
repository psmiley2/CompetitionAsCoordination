import globalVars as globs
# import evolution_continue_simulation as ev
import evolution as ev
import board as brd
import hyperParameters as hp

def main():
    board = brd.Board(hp.boardWidth, hp.boardHeight)
    evolution = ev.Evolution(board, "data/signals/r0.json")

    evolution.run()


if __name__ == "__main__":
    main()
