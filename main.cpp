#include <string>
#include <fstream>
#include <streambuf>

#include <xtensor/xnpy.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

#include "htmhelper.hpp"

constexpr int TOKEN_TYPE_NUM = 30;
constexpr int LEN_PER_TOKEN = 24;
constexpr int INPUT_SIZE = TOKEN_TYPE_NUM*LEN_PER_TOKEN;
constexpr int TP_DEPTH = 256;

std::vector<std::string> tokens = {".", "a", "b", "c", "ĉ", "d", "e", "f", "g", "ĝ", "h", "ĥ", "i", "j",
        "ĵ", "k", "l", "m", "n", "o", "p", "r", "s", "ŝ", "t", "u", "ŭ", "v", "z", " "};

std::string characterFromIndex(int index)
{
	return tokens[index];
}

xt::xarray<bool> encode(int token)
{
	xt::xarray<bool> base = xt::zeros<bool>({TOKEN_TYPE_NUM, LEN_PER_TOKEN});
	auto v = xt::view(base, token);
	v = true;
	return base;
}

xt::xarray<float> softmax(const xt::xarray<float>& x)
{
	auto b = xt::eval(xt::exp(x-xt::amax(x)));
	return b/xt::sum(b);
}

xt::xarray<float> categroize(const xt::xarray<bool>& in)
{
	xt::xarray<float> res = xt::zeros<float>({TOKEN_TYPE_NUM});
	assert(res.size()*LEN_PER_TOKEN == in.size());
	for(size_t i=0;i<in.size();i++)
		res[i/LEN_PER_TOKEN] += (float)in[i];
	res /= LEN_PER_TOKEN;
	return res;
}

xt::xarray<int> loadDataset(std::string path)
{
	xt::xarray<int> dataset = xt::load_npy<int>(path);
	return dataset;
}

struct Model
{
	Model(): tm({TOKEN_TYPE_NUM, LEN_PER_TOKEN}, TP_DEPTH, 2048, 8192)
	{
		tm->setMinThreshold(LEN_PER_TOKEN*0.35f+1);
		tm->setActivationThreshold(LEN_PER_TOKEN*0.75f);
		tm->setMaxNewSynapseCount(1024);
		tm->setPermanenceIncrement(0.055);
		tm->setPermanenceDecrement(0.055);
		tm->setConnectedPermanence(0.26);
		tm->setPredictedSegmentDecrement((1.f/TOKEN_TYPE_NUM)*tm->getPermanenceIncrement()*2.f);
		tm->setCheckInputs(false);
	}

	xt::xarray<bool> train(const xt::xarray<bool>& x)
	{
		return compute(x, true);
	}

	xt::xarray<bool> predict(const xt::xarray<bool>& x)
	{
		return compute(x, false);
	}

	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn)
	{
		return tm.compute(x, learn);
	}

	void reset()
	{
		tm.reset();
	}

	TM tm;
};

int main()
{
	auto dataset = loadDataset("dataset.npy");
	Model model;

	std::cout << "traning temporal memory..." << std::endl;
	for(int i=0;i<10;i++) {
		for(auto token : dataset)
			model.train(encode(token));
		model.reset();
		std::cout << i << "\r" << std::flush;
	}
	std::cout << "\n";

	std::cout << "Genetare random text....\n";
	model.reset();
	int token = 29;
	for(int i=0;i<400;i++) {
		auto res = model.predict(encode(token));
		auto prop = categroize(res);

		token = xt::argmax(prop)[0];
		std::cout << characterFromIndex(token);
	}

	std::cout << "\n\nFinishing a sentence....\n";
	model.reset();
	auto test = loadDataset("test.npy");
	xt::xarray<bool> res;
	for(auto token : test) {
		std::cout << characterFromIndex(token);
		res = model.predict(encode(token));
	}

	token = xt::argmax(categroize(res))[0];
	std::cout << characterFromIndex(token);

	for(int i=0;i<40 && token != 0;i++) {
		token = xt::argmax(categroize(model.predict(encode(token))))[0];
		std::cout << characterFromIndex(token);
	}
	std::cout << "\n";


	for(int i=0;i<TOKEN_TYPE_NUM;i++) {
		int token = i;
		std::cout << "\n\nGenerate random text started by " << characterFromIndex(i) << ":\n";
		xt::xarray<bool> res = encode(token);
		if(token < 0 or token >= TOKEN_TYPE_NUM)
			continue;
		model.reset();
		for(int i=0;i<80;i++) {
			std::cout << characterFromIndex(token);
			res = model.predict(encode(token));
			auto prop = categroize(res);

			token = xt::argmax(prop)[0];
			if(xt::sum(res)[0] == 0) {
				std::cout << "<NO PREDICTION>\n";
				break;
			}
			else if(token == 0) {
				std::cout << characterFromIndex(token);
				break;
			}
		}
	}
	
	std::cout << std::endl;
}
