class State(object):
	def __init__(self, h):
		self.g = 0
		self.h = h
		self.f = self.g+self.h
		self.tree = None
		self.search = 0
		self.pos = (0,0)
	def setG(self, new):
		self.g = new
		self.f = self.g + self.h
