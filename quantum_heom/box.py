"""Contains the Box class"""


class Box:

    """
    Creates a box object blah

    Parameters
    ----------
    source : str
        The type of source used
    dims : tuple of float
        The 2 dimensions of the box
    """


    def __init__(self, dims: tuple, **settings):

        # REQUIRED ATTRIBUTES
        if dims:
            self._dims = dims
        # SETTINGS
        if settings.get('source'):
            self._source = src
        else:
            self._source = 'sine'

    @property
    def source(self):

        return self._source

    @source.setter
    def source(self, src: str):

        # if src == 'sine':
        #     # function that does maths
        #     pass
        # elif src == 'cosine':
        #     # different one
        #     pass
        self._source = src

    @property
    def dims(self) -> tuple:

        return self._dims

    @dims.setter
    def dims(self, dims: tuple):

        self._dims = dims
