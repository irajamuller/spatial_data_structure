import pandas as pd
import numpy as np
import time
import math
import random
import heapq
import copy
import sys
import os
from io import StringIO
from collections import defaultdict
from typing import List, Tuple, Optional, Any

# ── dependências opcionais ──────────────────────────────────────────
from rtree import index as rtree_index

from scipy.spatial import KDTree as ScipyKDTree

from tabulate import tabulate

RUNS = 30   # execuções por benchmark
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

Point = Tuple[float, float]

# ═══════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ═══════════════════════════════════════════════════════════════════════

def euclidean(a: Point, b: Point) -> float:
	return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def timer(fn, *args, **kwargs):
	t0 = time.perf_counter()
	result = fn(*args, **kwargs)
	return time.perf_counter() - t0, result

def bench(fn, runs=RUNS, *args, **kwargs):
	times = []
	result = None
	for _ in range(runs):
		t, result = timer(fn, *args, **kwargs)
		times.append(t)
	return np.mean(times), np.std(times), result

# ═══════════════════════════════════════════════════════════════════════
# 1. QUADTREE
# ═══════════════════════════════════════════════════════════════════════

class _QTNode:
	__slots__ = ('cx','cy','hw','hh','cap','pts','nw','ne','sw','se','divided')
	def __init__(self, cx, cy, hw, hh, cap=16):
		self.cx=cx; self.cy=cy; self.hw=hw; self.hh=hh; self.cap=cap
		self.pts=[]; self.divided=False
		self.nw=self.ne=self.sw=self.se=None

	def _contains(self, x, y):
		return (self.cx-self.hw <= x <= self.cx+self.hw and
				self.cy-self.hh <= y <= self.cy+self.hh)

	def _intersects(self, x0,x1,y0,y1):
		return not (x0 > self.cx+self.hw or x1 < self.cx-self.hw or
					y0 > self.cy+self.hh or y1 < self.cy-self.hh)

	def _subdivide(self):
		hw2=self.hw/2; hh2=self.hh/2; cap=self.cap
		self.nw=_QTNode(self.cx-hw2,self.cy+hh2,hw2,hh2,cap)
		self.ne=_QTNode(self.cx+hw2,self.cy+hh2,hw2,hh2,cap)
		self.sw=_QTNode(self.cx-hw2,self.cy-hh2,hw2,hh2,cap)
		self.se=_QTNode(self.cx+hw2,self.cy-hh2,hw2,hh2,cap)
		self.divided=True

	def insert(self, x, y):
		if not self._contains(x, y): return False
		if len(self.pts) < self.cap:
			self.pts.append((x,y)); return True
		if not self.divided: self._subdivide()
		return (self.nw.insert(x,y) or self.ne.insert(x,y) or
				self.sw.insert(x,y) or self.se.insert(x,y))

	def query_exact(self, x, y):
		if not self._contains(x, y): return None
		for p in self.pts:
			if abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		if self.divided:
			for ch in (self.nw,self.ne,self.sw,self.se):
				r=ch.query_exact(x,y)
				if r: return r
		return None

	def query_rect(self, x0,x1,y0,y1, out):
		if not self._intersects(x0,x1,y0,y1): return
		for p in self.pts:
			if x0<=p[0]<=x1 and y0<=p[1]<=y1: out.append(p)
		if self.divided:
			for ch in (self.nw,self.ne,self.sw,self.se):
				ch.query_rect(x0,x1,y0,y1,out)

	def query_radius(self, cx, cy, r, out):
		# bounding-box pre-filter
		if not self._intersects(cx-r,cx+r,cy-r,cy+r): return
		for p in self.pts:
			if euclidean(p,(cx,cy))<=r: out.append(p)
		if self.divided:
			for ch in (self.nw,self.ne,self.sw,self.se):
				ch.query_radius(cx,cy,r,out)

	def nearest(self, x, y, best):
		dx=max(abs(x-self.cx)-self.hw,0)
		dy=max(abs(y-self.cy)-self.hh,0)
		if math.sqrt(dx*dx+dy*dy) >= best[0]: return best
		for p in self.pts:
			d=euclidean(p,(x,y))
			if d < best[0]: best=(d,p)
		if self.divided:
			for ch in (self.nw,self.ne,self.sw,self.se):
				best=ch.nearest(x,y,best)
		return best


class QuadTree:
	def __init__(self, cx, cy, hw, hh, cap=16):
		self.root=_QTNode(cx,cy,hw,hh,cap)

	def insert(self, x, y): self.root.insert(x, y)

	def query_exact(self, x, y): return self.root.query_exact(x, y)

	def query_rect(self, x0,x1,y0,y1):
		out=[]; self.root.query_rect(x0,x1,y0,y1,out); return out

	def query_radius(self, cx, cy, r):
		out=[]; self.root.query_radius(cx,cy,r,out); return out

	def nearest(self, x, y):
		res=self.root.nearest(x,y,(float('inf'),None))
		return res[1]


# ═══════════════════════════════════════════════════════════════════════
# 2. HASHING EXTENSÍVEL
# ═══════════════════════════════════════════════════════════════════════

class _EHBucket:
	__slots__ = ('depth','data','_cap')
	def __init__(self, depth, cap=64):
		self.depth=depth; self.data=[]; self._cap=cap

	def is_full(self): return len(self.data)>=self._cap

class ExtensibleHash:
	"""Hashing Extensível com chave (x,y) codificada como inteiro de 64 bits."""
	BUCKET_CAP = 64

	def __init__(self):
		self.global_depth = 1
		b0=_EHBucket(1,self.BUCKET_CAP)
		b1=_EHBucket(1,self.BUCKET_CAP)
		self.directory=[b0,b1]

	@staticmethod
	def _hash(x, y) -> int:
		# # Quantiza em grade de 1e-6 graus, mistura bits
		# ix = int(round(x * 1_000_000)) & 0xFFFFFFFF
		# iy = int(round(y * 1_000_000)) & 0xFFFFFFFF
		# h = ix ^ (iy * 2654435761)
		# h ^= (h >> 16)
		# h *= 0x45d9f3b
		# h &= 0xFFFFFFFF
		# h ^= (h >> 16)
		# return h
		return hash((round(x,6), round(y,6))) & 0xFFFFFFFF
	
	def _idx(self, h): return h & ((1 << self.global_depth) - 1)

	def insert(self, x, y):
		h=self._hash(x,y); 
		idx=self._idx(h)
		bucket=self.directory[idx]
		if not bucket.is_full():
			bucket.data.append((x,y))
			return
		if bucket.depth == self.global_depth:
			self.global_depth += 1
			self.directory = self.directory * 2   # dobra diretório
		self._split(bucket, h)

	def _split(self, bucket, h):
		old_data = bucket.data
		bucket.data = []
		bucket.depth += 1

		new_bucket = _EHBucket(bucket.depth, self.BUCKET_CAP)
		mask = 1 << (bucket.depth - 1)

		for i in range(len(self.directory)):
			if self.directory[i] is bucket:
				if i & mask:
					self.directory[i] = new_bucket
		for (x, y) in old_data:
			h = self._hash(x, y)
			idx = self._idx(h)
			self.directory[idx].data.append((x, y))

	def query_exact(self, x, y):
		h=self._hash(x,y); idx=self._idx(h)
		for p in self.directory[idx].data:
			if abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		return None

	def query_rect(self, x0,x1,y0,y1):
		seen=set(); out=[]
		for b in self.directory:
			if id(b) in seen: continue
			seen.add(id(b))
			for p in b.data:
				if x0<=p[0]<=x1 and y0<=p[1]<=y1: out.append(p)
		return out

	def query_radius(self, cx,cy,r):
		seen=set(); out=[]
		for b in self.directory:
			if id(b) in seen: continue
			seen.add(id(b))
			for p in b.data:
				if euclidean(p,(cx,cy))<=r: out.append(p)
		return out

	def nearest(self, x, y):
		seen=set(); best=(float('inf'),None)
		for b in self.directory:
			if id(b) in seen: continue
			seen.add(id(b))
			for p in b.data:
				d=euclidean(p,(x,y))
				if d<best[0]: best=(d,p)
		return best[1]


class Grid:
	"""Grade uniforme com células de tamanho fixo."""
	def __init__(self, x0,x1,y0,y1, rows=50, cols=50):
		self.x0=x0; self.y0=y0
		self.cw=(x1-x0)/cols; self.ch=(y1-y0)/rows
		self.rows=rows; self.cols=cols
		self.cells: dict = defaultdict(list)

	def _cell(self, x, y):
		ci=min(int((x-self.x0)/self.cw), self.cols-1)
		rj=min(int((y-self.y0)/self.ch), self.rows-1)
		return (ci,rj)

	def insert(self, x, y):
		self.cells[self._cell(x,y)].append((x,y))

	def query_exact(self, x, y):
		c=self._cell(x,y)
		for p in self.cells.get(c,[]):
			if abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		return None

	def query_rect(self, x0,x1,y0,y1):
		ci0=max(int((x0-self.x0)/self.cw),0)
		ci1=min(int((x1-self.x0)/self.cw),self.cols-1)
		rj0=max(int((y0-self.y0)/self.ch),0)
		rj1=min(int((y1-self.y0)/self.ch),self.rows-1)
		out=[]
		for ci in range(ci0,ci1+1):
			for rj in range(rj0,rj1+1):
				for p in self.cells.get((ci,rj),[]):
					if x0<=p[0]<=x1 and y0<=p[1]<=y1: out.append(p)
		return out

	def query_radius(self, cx, cy, r):
		ci0=max(int((cx-r-self.x0)/self.cw),0)
		ci1=min(int((cx+r-self.x0)/self.cw),self.cols-1)
		rj0=max(int((cy-r-self.y0)/self.ch),0)
		rj1=min(int((cy+r-self.y0)/self.ch),self.rows-1)
		out=[]
		for ci in range(ci0,ci1+1):
			for rj in range(rj0,rj1+1):
				for p in self.cells.get((ci,rj),[]):
					if euclidean(p,(cx,cy))<=r: out.append(p)
		return out

	def nearest(self, x, y):
		# Busca espiral de células
		best=(float('inf'),None)
		max_ring=max(self.rows,self.cols)
		ci0,rj0=self._cell(x,y)
		for ring in range(0,max_ring):
			min_d_cell=ring*min(self.cw,self.ch)
			if min_d_cell > best[0]: break
			for ci in range(max(ci0-ring,0),min(ci0+ring+1,self.cols)):
				for rj in range(max(rj0-ring,0),min(rj0+ring+1,self.rows)):
					if abs(ci-ci0)==ring or abs(rj-rj0)==ring:
						for p in self.cells.get((ci,rj),[]):
							d=euclidean(p,(x,y))
							if d<best[0]: best=(d,p)
		return best[1]


class _MTNode:
	__slots__ = ('pivot','radius','pts','children')
	def __init__(self):
		self.pivot=None; self.radius=0.0
		self.pts=[]; self.children=[]

_MT_LEAF_CAP = 32

def _mt_build(points, depth=0):
	node=_MTNode()
	if not points:
		return node
	# Escolhe pivot (ponto mais distante da média)
	cx=sum(p[0] for p in points)/len(points)
	cy=sum(p[1] for p in points)/len(points)
	pivot=max(points, key=lambda p: euclidean(p,(cx,cy)))
	node.pivot=pivot
	node.radius=max(euclidean(p,pivot) for p in points)

	if len(points)<=_MT_LEAF_CAP:
		node.pts=list(points)
		return node

	# Particiona em dois pelo pivot e o ponto mais distante
	far=max(points, key=lambda p: euclidean(p,pivot))
	g0=[p for p in points if euclidean(p,pivot)<=euclidean(p,far)]
	g1=[p for p in points if p not in g0]
	if not g0 or not g1:
		node.pts=list(points); return node
	node.children=[_mt_build(g0,depth+1), _mt_build(g1,depth+1)]
	return node

class MTree:
	def __init__(self, points):
		self._pts=[]
		self.root=_MTNode()

	def insert(self, x, y):
		self._pts.append((x,y))

	def _rebuild(self):
		self.root=_mt_build(self._pts)

	def query_exact(self, x, y):
		for p in self._pts:
			if abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		return None

	def _query_radius_node(self, node, cx, cy, r, out):
		if node.pivot is None: return
		if euclidean(node.pivot,(cx,cy)) - node.radius > r: return
		for p in node.pts:
			if euclidean(p,(cx,cy))<=r: out.append(p)
		for ch in node.children:
			self._query_radius_node(ch,cx,cy,r,out)

	def query_radius(self, cx, cy, r):
		out=[]; self._query_radius_node(self.root,cx,cy,r,out); return out

	def query_rect(self, x0,x1,y0,y1):
		cx=(x0+x1)/2; cy=(y0+y1)/2
		r=euclidean((x0,y0),(x1,y1))/2
		cands=self.query_radius(cx,cy,r)
		return [p for p in cands if x0<=p[0]<=x1 and y0<=p[1]<=y1]

	def _nearest_node(self, node, x, y, best):
		if node.pivot is None: return best
		d_pivot=euclidean(node.pivot,(x,y))
		if d_pivot - node.radius >= best[0]: return best
		for p in node.pts:
			d=euclidean(p,(x,y))
			if d<best[0]: best=(d,p)
		for ch in node.children:
			best=self._nearest_node(ch,x,y,best)
		return best

	def nearest(self, x, y):
		res=self._nearest_node(self.root,x,y,(float('inf'),None))
		return res[1]


class _KDNode:
	__slots__=('pt','left','right')
	def __init__(self,pt,left=None,right=None):
		self.pt=pt; self.left=left; self.right=right

def _kd_build(pts, depth=0):
	if not pts: return None
	axis=depth%2
	pts.sort(key=lambda p: p[axis])
	mid=len(pts)//2
	return _KDNode(pts[mid],
				   _kd_build(pts[:mid],depth+1),
				   _kd_build(pts[mid+1:],depth+1))

class KDTree:
	"""KD-tree 2D implementada do zero (fallback sem scipy)."""
	def __init__(self, points):
		self._pts=[]
		self.root=None

	def insert(self, x, y):
		self._pts.append((x,y))

	def _rebuild(self):
		self.root=_kd_build(list(self._pts))

	def _nn(self, node, x, y, depth, best):
		if node is None: return best
		d=euclidean(node.pt,(x,y))
		if d<best[0]: best=(d,node.pt)
		axis=depth%2
		diff=(x if axis==0 else y)-(node.pt[0] if axis==0 else node.pt[1])
		close,away=(node.left,node.right) if diff<=0 else (node.right,node.left)
		best=self._nn(close,x,y,depth+1,best)
		if abs(diff)<best[0]:
			best=self._nn(away,x,y,depth+1,best)
		return best

	def nearest(self, x, y):
		res=self._nn(self.root,x,y,0,(float('inf'),None))
		return res[1]

	def _range(self, node, x0,x1,y0,y1, depth, out):
		if node is None: return
		px,py=node.pt
		if x0<=px<=x1 and y0<=py<=y1: out.append(node.pt)
		axis=depth%2
		lo,hi=(x0,x1) if axis==0 else (y0,y1)
		v=px if axis==0 else py
		if lo<=v: self._range(node.left,x0,x1,y0,y1,depth+1,out)
		if v<=hi: self._range(node.right,x0,x1,y0,y1,depth+1,out)

	def query_rect(self, x0,x1,y0,y1):
		out=[]; self._range(self.root,x0,x1,y0,y1,0,out); return out

	def query_radius(self, cx,cy,r):
		out=[]
		self._range(self.root,cx-r,cx+r,cy-r,cy+r,0,out)
		return [p for p in out if euclidean(p,(cx,cy))<=r]

	def query_exact(self, x, y):
		res=self.nearest(x,y)
		if res and abs(res[0]-x)<1e-9 and abs(res[1]-y)<1e-9: return res
		return None


class KDTreeScipy:
	"""Wrapper sobre scipy.spatial.KDTree."""
	def __init__(self, points):
		self._pts=[]
		self._tree=None

	def insert(self, x, y):
		self._pts.append((x,y))

	def _rebuild(self):
		self._tree=ScipyKDTree(self._pts)

	def nearest(self, x, y):
		d,i=self._tree.query([x,y],k=1)
		return tuple(self._pts[i])

	def query_radius(self, cx, cy, r):
		idxs=self._tree.query_ball_point([cx,cy],r)
		return [tuple(self._pts[i]) for i in idxs]

	def query_rect(self, x0,x1,y0,y1):
		cx=(x0+x1)/2; cy=(y0+y1)/2
		r=max(x1-x0,y1-y0)
		cands=self.query_radius(cx,cy,r)
		return [p for p in cands if x0<=p[0]<=x1 and y0<=p[1]<=y1]

	def query_exact(self, x, y):
		d,i=self._tree.query([x,y],k=1)
		p=tuple(self._pts[i])
		return p if d<1e-9 else None


class RTree:
	"""Wrapper sobre a biblioteca rtree."""
	def __init__(self):
		p=rtree_index.Property()
		p.dimension=2
		self._idx=rtree_index.Index(properties=p)
		self._pts={}
		self._counter=0

	def insert(self, x, y):
		self._idx.insert(self._counter,(x,y,x,y))
		self._pts[self._counter]=(x,y)
		self._counter+=1

	def query_exact(self, x, y):
		eps=1e-9
		hits=list(self._idx.intersection((x-eps,y-eps,x+eps,y+eps)))
		for i in hits:
			p=self._pts[i]
			if abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		return None

	def query_rect(self, x0,x1,y0,y1):
		return [self._pts[i] for i in self._idx.intersection((x0,y0,x1,y1))]

	def query_radius(self, cx, cy, r):
		cands=[self._pts[i] for i in self._idx.intersection((cx-r,cy-r,cx+r,cy+r))]
		return [p for p in cands if euclidean(p,(cx,cy))<=r]

	def nearest(self, x, y):
		hits=list(self._idx.nearest((x,y,x,y),1))
		return self._pts[hits[0]] if hits else None


class _VPNode:
	__slots__=('vp','mu','left','right','pts')
	def __init__(self):
		self.vp=None; self.mu=0.0
		self.left=None; self.right=None; self.pts=[]

_VP_LEAF = 16

def _vp_build(pts):
	node=_VPNode()
	if not pts: return node
	if len(pts)<=_VP_LEAF:
		node.pts=list(pts); return node
	vp=pts[random.randrange(len(pts))]
	node.vp=vp
	dists=sorted(pts, key=lambda p: euclidean(p,vp))
	mid=len(dists)//2
	node.mu=euclidean(dists[mid],vp)
	node.left=_vp_build(dists[:mid])
	node.right=_vp_build(dists[mid:])
	return node

class VPTree:
	def __init__(self, points):
		self._pts=[]
		self.root=_VPNode()

	def insert(self, x, y):
		self._pts.append((x,y))

	def _rebuild(self):
		self.root=_vp_build(list(self._pts))

	def _nn_search(self, node, q, best):
		if node is None: return best
		for p in node.pts:
			d=euclidean(p,q)
			if d<best[0]: best=(d,p)
		if node.vp is None: return best
		d_vp=euclidean(node.vp,q)
		if d_vp<best[0]: best=(d_vp,node.vp)
		if d_vp<node.mu:
			best=self._nn_search(node.left,q,best)
			if d_vp+best[0]>=node.mu:
				best=self._nn_search(node.right,q,best)
		else:
			best=self._nn_search(node.right,q,best)
			if d_vp-best[0]<=node.mu:
				best=self._nn_search(node.left,q,best)
		return best

	def nearest(self, x, y):
		res=self._nn_search(self.root,(x,y),(float('inf'),None))
		return res[1]

	def _radius_search(self, node, cx, cy, r, out):
		if node is None: return
		for p in node.pts:
			if euclidean(p,(cx,cy))<=r: out.append(p)
		if node.vp is None: return
		d_vp=euclidean(node.vp,(cx,cy))
		if d_vp<=r: out.append(node.vp)
		if d_vp-r<=node.mu:
			self._radius_search(node.left,cx,cy,r,out)
		if d_vp+r>=node.mu:
			self._radius_search(node.right,cx,cy,r,out)

	def query_radius(self, cx, cy, r):
		out=[]; self._radius_search(self.root,cx,cy,r,out); return out

	def query_rect(self, x0,x1,y0,y1):
		cx=(x0+x1)/2; cy=(y0+y1)/2
		r=euclidean((x0,y0),(x1,y1))/2
		cands=self.query_radius(cx,cy,r)
		return [p for p in cands if x0<=p[0]<=x1 and y0<=p[1]<=y1]

	def query_exact(self, x, y):
		p=self.nearest(x,y)
		if p and abs(p[0]-x)<1e-9 and abs(p[1]-y)<1e-9: return p
		return None


def load_csv(path: str) -> List[Point]:
	df=pd.read_csv(path)
	lon=df['LONGITUDE'].astype(float).values
	lat=df['LATITUDE'].astype(float).values
	return list(zip(lon.tolist(), lat.tolist()))


def generate_sample_csv(path: str, n: int = 5000):
	"""Gera CSV de teste se nenhum arquivo for fornecido."""
	random.seed(SEED)
	lon0,lon1=-40.70,-40.55
	lat0,lat1=-19.55,-19.45
	rows=[]
	for i in range(n):
		lo=round(random.uniform(lon0,lon1),6)
		la=round(random.uniform(lat0,lat1),6)
		rows.append({'FID':i,'LONGITUDE':lo,'LATITUDE':la})
	pd.DataFrame(rows).to_csv(path,index=False)
	print(f"[INFO] CSV de exemplo gerado: {path} ({n} pontos)")


def make_query_params(points):
	"""Gera parâmetros de consulta representativos."""
	xs=[p[0] for p in points]; ys=[p[1] for p in points]
	xmin,xmax=min(xs),max(xs)
	ymin,ymax=min(ys),max(ys)
	# Ponto de consulta exata = ponto aleatório existente
	qp=random.choice(points)
	# Retângulo = ~5% da extensão total
	dx=(xmax-xmin)*0.05; dy=(ymax-ymin)*0.05
	cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
	rect=(cx-dx/2, cx+dx/2, cy-dy/2, cy+dy/2)  # x0,x1,y0,y1
	# Raio
	r=min(dx,dy)/2
	# Ponto de kNN (pode não estar no dataset)
	knn_pt=(cx,cy)
	return qp, rect, r, knn_pt


def run_benchmark(points: List[Point]):
	n=len(points)
	xs=[p[0] for p in points]; ys=[p[1] for p in points]
	xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
	pad=0.01
	cx=(xmin+xmax)/2; cy=(ymin+ymax)/2
	hw=(xmax-xmin)/2+pad; hh=(ymax-ymin)/2+pad

	qp, (rx0,rx1,ry0,ry1), rad, knn_pt = make_query_params(points)
	print(qp, (rx0,rx1,ry0,ry1), rad, knn_pt)
	# ── Fábricas ──────────────────────────────────────────────────────
	use_scipy = True

	structures = {}

	# Quadtree
	structures['Quadtree'] = {
		'build': lambda: (lambda qt: [qt.insert(x,y) for x,y in points] and qt)(
						  QuadTree(cx,cy,hw,hh,cap=16)),
		'exact':  None,
		'rect':   None,
		'radius': None,
		'nn':     None,
	}

	# Hashing Extensível
	structures['Ext.Hash'] = {
		'build': lambda: (lambda h: [h.insert(x,y) for x,y in points] and h)(ExtensibleHash()),
		'exact': None,'rect':None,'radius':None,'nn':None,
	}

	# Grid
	structures['Grid'] = {
		'build': lambda: (lambda g: [g.insert(x,y) for x,y in points] and g)(
						  Grid(xmin-pad,xmax+pad,ymin-pad,ymax+pad)),
		'exact':None,'rect':None,'radius':None,'nn':None,
	}

	# M-Tree
	structures['M-tree'] = {
		'build': lambda: (lambda m: [m.insert(x,y) for x,y in points] and m)(MTree(points)),
		'exact':None,'rect':None,'radius':None,'nn':None,
	}

	# KD-Tree
	structures['KD-tree'] = {
		'build': lambda: (lambda k: [k.insert(x,y) for x,y in points] and k)(KDTreeScipy(points)),
		'exact':None,'rect':None,'radius':None,'nn':None,
	}

	# R-Tree
	structures['R-tree'] = {
		'build': lambda: (lambda r: [r.insert(x,y) for x,y in points] and r)(RTree()),
		'exact':None,'rect':None,'radius':None,'nn':None,
	}

	# VP-Tree
	structures['VP-tree'] = {
		'build': lambda: (lambda v: [v.insert(x,y) for x,y in points] and v)(VPTree(points)),
		'exact':None,'rect':None,'radius':None,'nn':None,
	}

	# ── Benchmark de Inserção ──────────────────────────────────────────
	print("\n" + "="*70)
	print(f"  BENCHMARK  |  n = {n} pontos  |  {RUNS} execuções por operação")
	print("="*70)

	ins_results = {}
	built_structs = {}

	# print("\n[1/5] INSERÇÃO – construindo estrutura do zero a cada execução...")
	for name, s in structures.items():
		times=[]
		last=None
		for _ in range(RUNS):
			t0=time.perf_counter()
			last=s['build']()
			times.append(time.perf_counter()-t0)
		ins_results[name]=(np.mean(times)*1e3, np.std(times)*1e3)
		built_structs[name]=last
		print(f"   {name:12s}  média={ins_results[name][0]:.3f} ms  dp={ins_results[name][1]:.3f} ms")

	# Rebuild específico das estruturas que precisam (M-tree, KD, VP)
	# Garantir que o rebuild foi feito na última build
	for name in ['M-tree','KD-tree','VP-tree']:
		if name in built_structs:
			s=built_structs[name]
			if hasattr(s,'_rebuild'):
				s._rebuild()

	# ── Queries: usa estrutura já construída ───────────────────────────

	query_results = {name:{} for name in structures}

	def _bench_query(name, fn, label):
		times=[]
		res=None
		for _ in range(RUNS):
			t0=time.perf_counter()
			res=fn()
			times.append(time.perf_counter()-t0)
		query_results[name][label]=(np.mean(times)*1e6, np.std(times)*1e6, res)
		return res

	# Wrappers por estrutura
	def get_fns(name, s):
		if name=='Quadtree':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='Ext.Hash':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='Grid':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='M-tree':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='KD-tree':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='R-tree':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}
		if name=='VP-tree':
			return {
				'exact':  lambda: s.query_exact(*qp),
				'rect':   lambda: s.query_rect(rx0,rx1,ry0,ry1),
				'radius': lambda: s.query_radius(*knn_pt,rad),
				'nn':     lambda: s.nearest(*knn_pt),
			}

	queries_labels = [
		('exact',  '[2/5] CONSULTA EXATA'),
		('rect',   '[3/5] CONSULTA RETANGULAR'),
		('nn',     '[4/5] VIZINHO MAIS PRÓXIMO (kNN, k=1)'),
		('radius', '[5/5] CONSULTA POR RAIO'),
	]

	for qlabel, qprint in queries_labels:
		print(f"\n{qprint}")
		print(f"   Parâmetros: ", end="")
		if qlabel=='exact':
			print(f"ponto={qp}")
		elif qlabel=='rect':
			print(f"retângulo=[{rx0:.4f},{rx1:.4f}] x [{ry0:.4f},{ry1:.4f}]")
		elif qlabel=='nn':
			print(f"query={knn_pt}")
		else:
			print(f"centro={knn_pt}  raio={rad:.6f}")

		for name, s in built_structs.items():
			fns=get_fns(name,s)
			res=_bench_query(name, fns[qlabel], qlabel)
			mu,sd,_=query_results[name][qlabel]
			# Conta resultados
			cnt=len(res) if isinstance(res,list) else (1 if res else 0)
			print(f"   {name:12s}  média={mu:.2f} µs  dp={sd:.2f} µs  resultados={cnt}")

	# ── Tabela Resumo ──────────────────────────────────────────────────
	print("\n" + "="*70)
	print("  TABELA RESUMO  (média ± desvio padrão)")
	print("="*70)

	headers=["Estrutura","Inserção (ms)","Exata (µs)","Retangular (µs)","kNN (µs)","Raio (µs)"]
	rows=[]
	for name in structures:
		ins_m,ins_s=ins_results[name]
		def fmt(label):
			m,s,_=query_results[name][label]
			return f"{m:.2f}±{s:.2f}"
		rows.append([name,
					 f"{ins_m:.3f}±{ins_s:.3f}",
					 fmt('exact'), fmt('rect'), fmt('nn'), fmt('radius')])

	print(tabulate(rows, headers=headers, tablefmt='rounded_outline'))
	
	return ins_results, query_results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
	csv_path = 'amostra.csv'
	print(f"Lendo arquivo: {csv_path}")
	points = load_csv(csv_path)

	print(f"{len(points)} pontos carregados.")
	ins_res, qry_res = run_benchmark(points)