class vertex:
    x = 0.0
    y = 0.0
    z = 0.0

    triangles = []

    def __init__(self, xIn, yIn, zIn):
        self.x = xIn
        self.y = yIn
        self.z = zIn

    def __eq__(self, other):
        if (self.x == other.x and self.y == other.y and self.z == other.z):
            return True
        else:
            return False

    def __hash__(self):
        h1 = hash(self.x)
        h2 = hash(self.y)
        h3 = hash(self.z)
        return h1 + h2 + h3

    def add_triangle(self, triangle):
        self.triangles.append(triangle)

class vertex2d:
    x = 0.0
    y = 0.0

    lines = []

    def __init__(self, xIn, yIn):
        self.x = xIn
        self.y = yIn

    def __eq__(self, other):
        if (self.x == other.x and self.y == other.y):
            return True
        else:
            return False

    def __hash__(self):
        h1 = hash(self.x)
        h2 = hash(self.y)
        return h1 + h2

    def add_triangle(self, line):
        self.lines.append(line)

class triangle:
    v1 = vertex
    v2 = vertex
    v3 = vertex
    normal = vertex

    def __init__(self, v1In, v2In, v3In, normalIn):
        self.v1 = v1In
        self.v2 = v2In
        self.v3 = v3In
        v1In.add_triangle(self)
        v2In.add_triangle(self)
        v3In.add_triangle(self)
        self.normal = normalIn

    def add_vertex(self, vert):
        if self.v1.__eq__(vert):
            self.v1 = vert
        if self.v2.__eq__(vert):
            self.v2 = vert
        if self.v3.__eq__(vert):
            self.v3 = vert


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
    x_max = 0.0
    x_min = 0.0
    y_max = 0.0
    y_min = 0.0

    def __init__(self, linesIn):
        self.lines = linesIn
        for line in linesIn:
            self.x_max = max(line.x1, self.x_max)
            self.x_min = min(line.x1, self.x_min)
            self.y_max = max(line.y1, self.y_max)
            self.y_min = min(line.y1, self.y_min)
