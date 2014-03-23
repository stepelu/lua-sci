--------------------------------------------------------------------------------
-- Statistical functions module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

-- Variances and covariances are computed according to the unbiased version of
-- the algorithm.
-- Welford-type algorithms are used for superior numerical stability, see:
-- http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
-- http://www.johndcook.com/standard_deviation.html

-- TODO: BIC, AIC.
-- TODO: Function combine(...) to join results for parallel computing.

-- TODO: Speed-up via OpenBLAS?
-- TODO: Speed-up via FFI cdata.

local alg  = require "sci.alg" 
local dist = require "sci.dist"
local quad = require "sci.quad"

local sqrt, abs, max = math.sqrt, math.abs, math.max
local vec, mat, diag = alg.vec, alg.mat, alg.diag
local type = type

local function mean(x)
  if #x < 1 then
    error("#x >=1 required: #x="..#x)
  end
  local mu = 0
  for i=1,#x do
    mu = mu + (x[i] - mu)/i
  end
  return mu
end

local function var(x)
  if #x < 2 then
    error("#x >= 2 required: #x"..#x)
  end
  local mu, s2 = 0, 0
  for i=1,#x do
    local delta = x[i] - mu
    mu = mu + delta/i
    s2 = s2 + delta*(x[i] - mu)
  end
  return s2/(#x - 1)
end

local function cov(x, y)
  local mux, muy, s2c = 0, 0, 0
  if not #x == #y then
    error("#x ~= #y: #x="..#x..", #y="..#y)
  end
  if #x < 2 then
    error("#x >= 2 required: #x="..#x)
  end
  for i=1,#x do
    local deltax = x[i] - mux
    local deltay = y[i] - muy
    local r = 1/i
    mux = mux + deltax*r
    muy = muy + deltay*r
    s2c = s2c + deltax*(y[i] - muy)
  end
  return s2c/(#x - 1)
end

local function cor(x, y)
  return cov(x, y)/sqrt(var(x)*var(y))
end

local mean_mt = {
  clear = function(self)
    self._n = 0
    self._mu:fill(0)
  end,
  push = function(self, x)
    self._n = self._n + 1
    self._mu:set(self._mu + (x - self._mu)/self._n)    
  end,
  mean = function(self, mean)
    if self._n < 1 then
      error("n >= 1 required: n="..self._n)
    end
    mean:set(self._mu)
  end,
}
mean_mt.__index = mean_mt

local mean0_mt = {
  clear = function(self)
    self._n = 0
    self._mu = 0
  end,
  push = function(self, x)
    self._n = self._n + 1
    self._mu = self._mu + (x - self._mu)/self._n
  end,
  mean = function(self)
    return self._mu
  end,
}
mean0_mt.__index = mean0_mt

local var_mt = {
  clear = function(self)
    self._n = 0
    self._mu:fill(0)
    self._s2:fill(0)
  end,
  push = function(self, x)
    self._n = self._n + 1
    local r = 1/self._n
    self._delta:set(x - self._mu)
    self._mu:set(self._mu + self._delta*r)
    self._s2:set(self._s2 + self._delta*(x - self._mu))
  end,
  mean = mean_mt.mean,
  var = function(self, var)
    if self._n < 2 then
      error("n >= 2 required: n="..self._n)
    end
    var:set(self._s2/(self._n - 1))
  end,
}
var_mt.__index = var_mt

local var0_mt = {
  clear = function(self)
    self._n = 0
    self._mu = 0
    self._s2 = 0
  end,
  push = function(self, x)
    self._n = self._n + 1
    local r = 1/self._n
    self._delta = x - self._mu
    self._mu = self._mu + self._delta*r
    self._s2 = self._s2 + self._delta*(x - self._mu)
  end,
  mean = mean0_mt.mean,
  var = function(self)
    if self._n < 2 then
      error("n >= 2 required: n="..self._n)
    end
    return self._s2/(self._n - 1)
  end,
}
var0_mt.__index = var0_mt

-- Y *can* alias X.
local function covtocor(X, Y)
  local n, m = X:nrow(), X:ncol()
  local ny, my = Y:nrow(), Y:ncol()
  if not (n == m and n == ny and n == my) then
    error("dimensions of X and Y must agree: X is "..n.."x"..m
        ..", Y is "..ny.."x"..my)
  end
  for r=1,n do
    for c=1,n do
      if r ~= c then
        Y[r][c] = X[r][c]/sqrt(X[r][r]*X[c][c])
      end
    end
  end
  Y:fill(1, diag())
end

local cov_mt = {
  clear = function(self)
    self._n = 0
    self._mu:fill(0)
    self._s2:fill(0)
  end,
  push = function(self, x)
    self._n = self._n + 1
    local r = 1/self._n
    local delta, mu, s2 = self._delta, self._mu, self._s2
    delta:set(x - mu)
    mu:set(mu + delta*r)
    for i=1,#delta do for j=1,#delta do
      s2[i][j] = s2[i][j] + delta[i]*(x[j] - mu[j])
    end end
  end,
  mean = mean_mt.mean,
  var = function(self, var)
    if self._n < 2 then
      error("n >= 2 required: n="..self._n)
    end
    var:set(self._s2/(self._n - 1), diag())
  end,
  cov = function(self, cov)
    if self._n < 2 then
      error("n >= 2 required: n="..self._n)
    end
    cov:set(self._s2/(self._n - 1))
  end,
  cor = function(self, cor)
    self:cov(cor)
    covtocor(cor, cor)
  end
}
cov_mt.__index = cov_mt

local function olmean(dim)
  if dim == 0 then
    return setmetatable({ _n = 0, _mu = 0 }, mean0_mt)
  else
    return setmetatable({ _n = 0, _mu = vec(dim) }, mean_mt)
  end
end
local function olvar(dim)
  if dim == 0 then
    return setmetatable({ _n = 0, _mu = 0, _delta = 0, _s2 = 0 }, var0_mt)
  else
    return setmetatable({ _n = 0, _mu = vec(dim), _delta = vec(dim), 
      _s2 = vec(dim) }, var_mt)
  end
end
local function olcov(dim)
    return setmetatable({ _n = 0, _mu = vec(dim), _delta = vec(dim), 
      _s2 = mat(dim, dim) }, cov_mt)
end

return {
  mean = mean,
  var  = var,
  cov  = cov,
  cor  = cor,
  
  olmean = olmean,
  olvar  = olvar, 
  olcov  = olcov,
}
