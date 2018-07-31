#pragma once

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/Anomaly.hpp>

#include <xtensor/xarray.hpp>

using namespace nupic;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::temporal_memory::TemporalMemory;
using nupic::algorithms::Cells4::Cells4;
using nupic::algorithms::anomaly::Anomaly;

inline std::vector<UInt> sparsify(const xt::xarray<bool>& t)
{
	std::vector<UInt> v;
	v.reserve(t.size()/10);
	for(size_t i=0;i<t.size();i++)
	{
		if(t[i])
			v.push_back(i);
	}
	return v;
}

template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

struct HTMLayerBase
{
	HTMLayerBase() = default;
	HTMLayerBase(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: inputDimentions(inDim), outputDimentions(outDim){}
	
	std::vector<size_t> inputDimentions;
	std::vector<size_t> outputDimentions;
	
	size_t inputSize()
	{
		size_t s = 1;
		for(auto v : inputDimentions)
			s *= v;
		return s;
	}
	
	size_t outputSize()
	{
		size_t s = 1;
		for(auto v : outputDimentions)
			s *= v;
		return s;
	}
};

struct SP : HTMLayerBase
{
	SP() = default;
	SP(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: HTMLayerBase(inDim, outDim)
	{
		std::vector<UInt> inSize = as<std::vector<UInt>>(inputDimentions);
		std::vector<UInt> outSize = as<std::vector<UInt>>(outputDimentions);
		sp = SpatialPooler(inSize, outSize);
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<UInt> in(inputSize());
		std::vector<UInt> out(outputSize());
		for(size_t i=0;i<t.size();i++)
			in[i] = t[i];
		
		sp.compute(in.data(), learn, out.data());
		
		xt::xarray<bool> res = xt::zeros<bool>(outputDimentions);
		for(size_t i=0;i<out.size();i++)
			res[i] = out[i];
		return res;
	}

	SpatialPooler* operator-> ()
	{
		return &sp;
	}

	const SpatialPooler* operator-> () const
	{
		return &sp;
	}
	
	SpatialPooler sp;
};

struct TP : HTMLayerBase
{
	TP() = default;
	TP(std::vector<size_t> inDim, size_t numCol)
		: HTMLayerBase(inDim, inDim), colInTP(numCol)
		, tp(inputSize(), colInTP, 12, 8, 15, 5, .5, .8, 1.0, .1, .1, 0.0,
			false, 42, true, false)
	{
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<Real> in(t.size());
		std::vector<Real> out(t.size()*colInTP);
		for(size_t i=0;i<t.size();i++)
			in[i] = t[i];
		tp.compute(in.data(), out.data(), true, learn);
		xt::xarray<bool> res = xt::zeros<bool>(outputDimentions);
		for(size_t i=0;i<out.size()/colInTP;i++)
			res[i] = out[i*colInTP];
		return res;
	}

	Cells4* operator-> ()
	{
		return &tp;
	}

	const Cells4* operator-> () const
	{
		return &tp;
	}
	
	void reset()
	{
		tp.reset();
	}
	
	size_t colInTP;
	Cells4 tp;
};

struct TM : HTMLayerBase
{
	TM() = default;
	TM(std::vector<size_t> inDim, size_t numCol, size_t maxSegmentsPerCell=255, size_t maxSynapsesPerSegment=255)
		: HTMLayerBase(inDim, inDim), colInTP(numCol)
	{
		std::vector<UInt> inSize = as<std::vector<UInt>>(inputDimentions);
		tm = TemporalMemory(inSize, numCol, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, maxSegmentsPerCell, maxSynapsesPerSegment, true);
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<UInt> cols = sparsify(t);
		xt::xarray<bool> tpOutput = xt::zeros<bool>(t.shape());
		tm.compute(cols.size(), &cols[0], learn);
		auto next = tm.getPredictiveCells();
		for(size_t i=0;i<next.size();i++)
		{
			int idx = next[i]/colInTP;
			tpOutput[idx] = true;
		}
		return tpOutput;
	}

	TemporalMemory* operator-> ()
	{
		return &tm;
	}

	const TemporalMemory* operator-> () const
	{
		return &tm;
	}
	
	void reset()
	{
		tm.reset();
	}
	
	size_t colInTP;
	TemporalMemory tm;
};
