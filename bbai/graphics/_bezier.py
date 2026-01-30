import numpy as np

from .._computation._bridge import visf

class BezierPath:
    """ Uses an orthogonal distance fitting algorithm to fit a Bezier path to a graph or
    a parametric function.

    The path can be either used directly or accessed as a tikz string.

    Parameters
    ----------
    dst_xmin : float, default=-6
        The min x value for the rendered path.

    dst_xmax : float, default=-6
        The max x value for the rendered path.

    dst_ymin : float, default=-6
        The min y value for the rendered path.

    dst_ymax : float, default=-6
        The max y value for the rendered path.

    src_xmin : float, default=np.nan
        The path will be rendered so that src_xmin maps to dst_xmin. By
        default src_xmin will be set to the functions min x value.

    src_xmax : float, default=-6
        The path will be rendered so that src_xmax maps to dst_xmax. By
        default src_xmax will be set to the functions max x value.

    src_ymin : float, default=-6
        The path will be rendered so that src_ymin maps to dst_ymin. By
        default src_xmin will be set to the functions min y value.

    src_ymax : float, default=-6
        The path will be rendered so that src_ymax maps to dst_ymax. By
        default src_ymax will be set to the functions max y value.

    max_distance : float, default=1.0e-2
        The fitting algorithm will continue until either it reaches max_segments
        or the maximum of the minimal orthogonal distance from the Bezier path to
        the target curve is less than max_distance.

    max_segments : int, default=10000
        The maximal number of segments for the Bezier path.

    Examples
    --------
    >>> from bbai.graphics import BezierPath
    >>> def f(t):
    >>>   return np.exp(-t*t)
    >>> path = BezierPath().fit(f, -1, 1) # fit Gaussian over the range [-1, 1]
    >>> print(path.tikz_) # returns the tikz representation of the fitted path
        
    """
    def __init__(self,  
            dst_xmin=-6, dst_xmax=6, 
            dst_ymin=0, dst_ymax=1,
            src_xmin=np.nan, src_xmax=np.nan,
            src_ymin=np.nan, src_ymax=np.nan,
            max_distance = 1.0e-2,
            max_segments = 10000,
        ):
        self.opts_ = dict(
          max_distance = max_distance,
          max_segments = max_segments,
        
          source_xmin = src_xmin,
          source_xmax = src_xmax,
          source_ymin = src_ymin,
          source_ymax = src_ymax,
        
          target_xmin = dst_xmin,
          target_xmax = dst_xmax,
          target_ymin = dst_ymin,
          target_ymax = dst_ymax,
        )

    def fit(self, *args):
        """Fit a bezier path to the provided function.

           If only one function is provided, it will fit
           a graph; if two are provided, it will fit a 
           parametric function.
        """
        nargs = len(args)
        assert nargs == 3 or nargs == 4
        if nargs == 3:
            self._fit1(*args)
        else:
            self._fit2(*args)
        return self

    def _fit1(self, f, a, b):
        opts = dict(self.opts_)
        opts['a'] = a
        opts['b'] = b
        def fp(xs):
            try:
                return f(xs)
            except Exception as e:
                err = e
                return np.zeros(0)
        res = visf.fit_bezier_path(fp, opts)
        if res['max_distance'] < 0:
            raise err

        self._set_path(res)

    def _fit2(self, fx, fy, a, b):
        opts = dict(self.opts_)
        opts['a'] = a
        opts['b'] = b
        def fxp(xs):
            try:
                return fx(xs)
            except Exception as e:
                err = e
                return np.zeros(0)
        def fyp(xs):
            try:
                return fy(xs)
            except Exception as e:
                err = e
                return np.zeros(0)
        res = visf.fit_parametric_bezier_path(fxp, fyp, opts)
        if res['max_distance'] < 0:
            raise err

        self._set_path(res)


    def _set_path(self, res):
        self.segments_ = res['path'].T

        self.max_distance_ = res['max_distance']
        self.max_distance_x_ = res['max_distance_x']
        self.max_distance_y_ = res['max_distance_y']
        self.max_distance_t_ = res['max_distance_t']

        self.src_xmin_ = res['source_xmin']
        self.src_xmax_ = res['source_xmax']
        self.src_ymin_ = res['source_ymin']
        self.src_ymax_ = res['source_ymax']

        self.dst_xmin_ = res['target_xmin']
        self.dst_xmax_ = res['target_xmax']
        self.dst_ymin_ = res['target_ymin']
        self.dst_ymax_ = res['target_ymax']

        self.tikz_ = res['tikz']
