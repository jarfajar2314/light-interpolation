import numpy as np
import scipy.linalg
from scipy.spatial.distance import pdist, cdist
from scipy.optimize import least_squares
from . import variogram_models, utils


class OrdinaryKriging:
    def __init__(self, x, y, z, variogram_model="exponential"):
        self.weight = False
        self.pseudo_inv = False
        self.eps = 1e-10
        self.nlags = 6
        self.variogram_model = variogram_model
        self.XCENTER = 0.0
        self.YCENTER = 0.0

        self.variogram_dict = {
            "power": variogram_models.power_variogram_model,
            "gaussian": variogram_models.gaussian_variogram_model,
            "spherical": variogram_models.spherical_variogram_model,
            "exponential": variogram_models.exponential_variogram_model,
        }

        self.variogram_function = self.variogram_dict[self.variogram_model]

        self.X_ADJUSTED = np.atleast_1d(
            np.squeeze(np.array(x, copy=True, dtype=np.float64))
        )
        self.Y_ADJUSTED = np.atleast_1d(
            np.squeeze(np.array(y, copy=True, dtype=np.float64))
        )
        self.Z = np.atleast_1d(np.squeeze(
            np.array(z, copy=True, dtype=np.float64)))

        print("Initializing variogram model...")
        (
            self.lags,
            self.semivariance,
            self.variogram_model_parameters
        ) = self.initialize_variogram(
            np.vstack([self.X_ADJUSTED, self.Y_ADJUSTED]).T,
            self.Z,
            self.weight
        )

        self.sill = self.variogram_model_parameters[0] + \
            self.variogram_model_parameters[2]
        self.range_a = self.variogram_model_parameters[1]
        self.nugget = self.variogram_model_parameters[2]

        print(" Partial Sill:", self.variogram_model_parameters[0])
        print(
            " Full Sill:",
            self.variogram_model_parameters[0]
            + self.variogram_model_parameters[2],
        )
        print(" Range:", self.variogram_model_parameters[1])
        print(" Nugget:", self.variogram_model_parameters[2], "\n")

    def initialize_variogram(self, X, y, weight):
        if X.shape[1] != 2:
            raise ValueError(
                "Geographic coordinate type only supported for 2D datasets."
            )
        x1, x2 = np.meshgrid(X[:, 0], X[:, 0], sparse=True)
        y1, y2 = np.meshgrid(X[:, 1], X[:, 1], sparse=True)
        z1, z2 = np.meshgrid(y, y, sparse=True)
        d = utils.great_circle_distance(x1, y1, x2, y2)
        g = 0.5 * (z1 - z2) ** 2.0
        indices = np.indices(d.shape)
        d = d[(indices[0, :, :] > indices[1, :, :])]
        g = g[(indices[0, :, :] > indices[1, :, :])]

        # Grouping points by distance
        dmax = np.amax(d)
        dmin = np.amin(d)
        dd = (dmax - dmin) / self.nlags
        bins = [dmin + n * dd for n in range(self.nlags)]
        dmax += 0.001
        bins.append(dmax)

        lags = np.zeros(self.nlags)
        semivariance = np.zeros(self.nlags)

        # calculate mean of distance and semivariance of each group
        for n in range(self.nlags):
            if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
                lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
                semivariance[n] = np.mean(
                    g[(d >= bins[n]) & (d < bins[n + 1])])
            else:
                lags[n] = np.nan
                semivariance[n] = np.nan

        lags = lags[~np.isnan(semivariance)]
        semivariance = semivariance[~np.isnan(semivariance)]

        print(" lags:", lags)
        print(" semivariance:", semivariance)

        variogram_model_parameters = self.calculate_variogram_model(
            lags, semivariance, self.variogram_model, self.variogram_function, weight
        )

        return lags, semivariance, variogram_model_parameters

    def calculate_variogram_model(self, lags, semivariance, variogram_model, variogram_function, weight):
        print("Calculating variogram model...")
        print(" Using '%s' Variogram Model" % variogram_model)
        if variogram_model == "power":
            x0 = [
                (np.amax(semivariance) - np.amin(semivariance))
                / (np.amax(lags) - np.amin(lags)),
                1.1,
                np.amin(semivariance),
            ]
            bnds = ([0.0, 0.001, 0.0], [np.inf, 1.999, np.amax(semivariance)])
        else:
            x0 = [
                np.amax(semivariance) - np.amin(semivariance),
                0.25 * np.amax(lags),
                np.amin(semivariance),
            ]
            bnds = (
                [0.0, 0.0, 0.0],
                [10.0 * np.amax(semivariance), np.amax(lags),
                 np.amax(semivariance)],
            )

        print(" x0:", x0)
        print(" bnds:", bnds)
        # use 'soft' L1-norm minimization in order to buffer against
        # potential outliers (weird/skewed points)
        print("Fitting variogram model...")
        res = least_squares(
            self.variogram_residuals,
            x0,
            bounds=bnds,
            loss="soft_l1",
            args=(lags, semivariance, variogram_function, weight),
        )

        return res.x

    def get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""
        d = utils.great_circle_distance(
            self.X_ADJUSTED[:, np.newaxis],
            self.Y_ADJUSTED[:, np.newaxis],
            self.X_ADJUSTED,
            self.Y_ADJUSTED,
        )
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = - \
            self.variogram_function(self.variogram_model_parameters, d)

        np.fill_diagonal(a, 0.0)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0
        return a

    def exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        # use the desired method to invert the kriging matrix
        if self.pseudo_inv:
            a_inv = scipy.linalg.pinv(a)
        else:
            a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n + 1, 1))
        b[:, :n, 0] = - \
            self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value and self.exact_values:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n + 1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n + 1)).T).reshape((1, n + 1, npt)).T
        zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    def variogram_residuals(self, params, x, y, variogram_function, weight):
        if weight:
            drange = np.amax(x) - np.amin(x)
            k = 2.1972 / (0.1 * drange)
            x0 = 0.7 * drange + np.amin(x)
            weights = 1.0 / (1.0 + np.exp(-k * (x0 - x)))
            weights /= np.sum(weights)
            resid = (variogram_function(params, x) - y) * weights
        else:
            resid = variogram_function(params, x) - y

        return resid

    def predict(self, x_new, y_new, style="grid",):
        xpts = np.atleast_1d(np.squeeze(np.array(x_new, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(y_new, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        a = self.get_kriging_matrix(n)

        if style == "grid":
            npt = ny * nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()
        elif style == "points":
            if xpts.size != ypts.size:
                raise ValueError(
                    "xpoints and ypoints must have "
                    "same dimensions when treated as "
                    "listing discrete points."
                )
            npt = nx
        else:
            raise ValueError(
                "style argument must be 'grid', or 'points'")

        mask = np.zeros(npt, dtype="bool")

        bd = utils.great_circle_distance(
            xpts[:, np.newaxis],
            ypts[:, np.newaxis],
            self.X_ADJUSTED,
            self.Y_ADJUSTED,
        )

        zvalues, sigmasq = self.exec_vector(a, bd, mask)

        if (style == "points"):
            zvalues = zvalues[0]
            sigmasq = sigmasq[0]
        elif (style == "grid"):
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq
