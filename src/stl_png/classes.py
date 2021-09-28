import sys


class vertex:
    x = 0.0
    y = 0.0
    z = 0.0

    def __init__(self, xIn, yIn, zIn):
        self.x = xIn
        self.y = yIn
        self.z = zIn


class triangle:
    v1 = vertex
    v2 = vertex
    v3 = vertex
    normal = vertex

    def __init__(self, v1In, v2In, v3In, normalIn):
        self.v1 = v1In
        self.v2 = v2In
        self.v3 = v3In
        self.normal = normalIn


class line:
    x1 = 0.0
    y1 = 0.0
    x2 = 0.0
    y2 = 0.0
    normx = 0.0
    normy = 0.0

    def __init__(self, x1In, y1In, x2In, y2In, normxIn, normyIn):
        self.x1 = x1In
        self.y1 = y1In
        self.x2 = x2In
        self.y2 = y2In
        self.normx = normxIn
        self.normy = normyIn


class figure:
    lines = []
    x_max = -sys.float_info.max
    x_min = sys.float_info.max
    y_max = -sys.float_info.max
    y_min = sys.float_info.max

    def __init__(self, linesIn):
        self.lines = linesIn
        for line in linesIn:
            self.x_max = max(line.x1, self.x_max)
            self.x_min = min(line.x1, self.x_min)
            self.y_max = max(line.y1, self.y_max)
            self.y_min = min(line.y1, self.y_min)