import globalVars as globs
import hyperParameters as hp

class Fitness:
    def __init__(self, board):
        self.board = board
        self.totalScore = 0
        self.aspectRatioScore = 0
        self.cellRatioScore = 0
        self.sizeScore = 0

        self.topCell = 1000000
        self.bottomCell = -1000000
        self.rightCell = -1000000
        self.leftCell = 1000000
        self.stemCellCount = 0
        self.nerveCellCount = 0
        self.leftNerveCellCount = 0
        self.rightNerveCellCount = 0
        self.leftSkinCellCount = 0
        self.rightSkinCellCount = 0
        self.skinCellCount = 0
        for i in range(len(self.board.grid)):
            for j in range(len(self.board.grid[i])):              
                if self.board.grid[i][j] is not None:
                    self.topCell = min(self.topCell, i)
                    self.bottomCell = max(self.bottomCell, i)
                    self.leftCell = min(self.leftCell, j)
                    self.rightCell = max(self.rightCell, j)

        self.height = self.bottomCell - self.topCell + 1
        self.width = self.rightCell - self.leftCell + 1
        self.midWidth = self.leftCell + int((self.rightCell - self.leftCell) / 2)
        self.midHeight = self.topCell + int((self.bottomCell - self.topCell) / 2)

        for i in range(len(self.board.grid)):
            for j in range(len(self.board.grid[i])):  
                if self.board.grid[i][j] is None:
                    continue

                if self.board.grid[i][j].type == globs.STEM:
                    self.stemCellCount += 1
                elif self.board.grid[i][j].type == globs.NERVE:
                    self.nerveCellCount += 1
                    if j < self.midWidth:
                        self.leftNerveCellCount += 1
                    else:
                        self.rightNerveCellCount += 1
                elif self.board.grid[i][j].type == globs.SKIN:
                    self.skinCellCount += 1
                    if j < self.midWidth:
                        self.leftSkinCellCount += 1
                    else:
                        self.rightSkinCellCount += 1

        cellValues = [globs.STEM, globs.NERVE, globs.SKIN]
        self.startCol = -1 
        self.endCol = -1
        for i in range(len(self.board.grid)):
            if self.board.grid[i][self.midWidth] is None:
                continue

            if self.board.grid[i][self.midWidth].type in cellValues:
                if self.startCol == -1:
                    self.startCol = i
                self.endCol = i

        self.startRow = -1
        self.endRow = -1
        for i in range(len(self.board.grid[0])):
            if self.board.grid[self.midHeight][i] is None:
                continue

            if self.board.grid[self.midHeight][i].type in cellValues:
                if self.startRow == -1:
                    self.startRow = i
                self.endRow = i
                


    def calculate(self):
        totalCellCount = self.stemCellCount + self.skinCellCount + self.nerveCellCount

        # Aspect Ratio
        realAspectRatio = self.width / self.height
        targetAspectRatio = hp.targetAspectRatio 
        self.aspectRatioScore = -50 * abs((realAspectRatio / targetAspectRatio) - 1) + 50
        if self.aspectRatioScore < 0:
            self.aspectRatioScore = 0

        if hp.useMidpointsForAspectRatio:
            w = (self.endRow - self.startRow) + 1
            h = (self.endCol - self.startCol) + 1
            realAspectRatio =  w / h
            self.aspectRatioScore = -50 * abs((realAspectRatio / targetAspectRatio) - 1) + 50
            if self.width > w:
                self.aspectRatioScore -= self.width - w
            if self.height > h:
                self.aspectRatioScore -= self.height - h
            if self.aspectRatioScore < 0:
                self.aspectRatioScore = 0

        # Cell Ratio Left Side -> want more nerve cells 
        if self.leftNerveCellCount == 0:
            self.leftNerveCellCount = 1
        leftRealCellRatio = self.leftSkinCellCount / self.leftNerveCellCount
        leftCellRatioScore = -25 * abs(leftRealCellRatio / 3) + 25
        if self.leftNerveCellCount == 0:
            leftCellRatioScore = 0

        # Cell Ratio Right Side -> want more skin cells 
        if self.rightSkinCellCount == 0:
            self.rightSkinCellCount = 1
        rightRealCellRatio = self.rightNerveCellCount / self.rightSkinCellCount
        rightCellRatioScore = -25 * abs(rightRealCellRatio / 3) + 25
        if self.rightSkinCellCount == 0:
            rightCellRatioScore = 0

        self.cellRatioScore = leftCellRatioScore + rightCellRatioScore
        if self.cellRatioScore < 0:
            self.cellRatioScore = 0

        self.sizeScore = -50 * abs((totalCellCount / hp.targetSize) - 1) + 50
        if self.sizeScore < 0:
            self.sizeScore = 0

        self.totalScore = self.aspectRatioScore + self.cellRatioScore + self.sizeScore
        

        if hp.onlyUseSizeAsFitness:
            self.totalScore = -100 * abs((totalCellCount / hp.targetSize) - 1) + 100
        
        if totalCellCount < 50:
            self.totalScore = 0

        if hp.punishForHittingBorder:
            if self.topCell <= 1 or self.bottomCell >= (hp.boardHeight - 2) or self.leftCell <= 1 or self.rightCell >= (hp.boardWidth - 2):
                self.totalScore = 0 
        
        if self.totalScore < 0:
            self.totalScore = 0
            

    def getTotalScore(self): 
        return self.totalScore
