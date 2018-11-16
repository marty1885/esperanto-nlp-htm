#include <string>
#include <fstream>
#include <streambuf>

#include <xtensor/xnpy.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xindex_view.hpp>

#include "HTMHelper.hpp"
using namespace HTM;

constexpr int TOKEN_TYPE_NUM = 30;
constexpr int LEN_PER_TOKEN = 16;
constexpr int INPUT_SIZE = TOKEN_TYPE_NUM*LEN_PER_TOKEN;
constexpr int TP_DEPTH = 1024;

std::vector<std::string> lower_esperanto_chars = {".", "a", "b", "c", "ĉ", "d", "e", "f", "g", "ĝ", "h", "ĥ", "i", "j",
        "ĵ", "k", "l", "m", "n", "o", "p", "r", "s", "ŝ", "t", "u", "ŭ", "v", "z", " "};
std::vector<std::string> upper_esperanto_chars = {".", "A", "B", "C", "Ĉ", "D", "E", "F", "G", "Ĝ", "H", "Ĥ", "I", "J",
	"Ĵ", "K", "L", "M", "N", "O", "P", "R", "S", "Ŝ", "T", "U", "Ŭ", "V", "Z", " "};

std::string characterFromIndex(int index, bool upper=false)
{
	if(upper)
		return upper_esperanto_chars[index];
	return lower_esperanto_chars[index];}

xt::xarray<bool> encode(int token)
{
	CategoryEncoder encoder(TOKEN_TYPE_NUM, LEN_PER_TOKEN);
	return encoder(token);
}

xt::xarray<float> linearprop(const xt::xarray<float>& x)
{
	auto s = xt::sum(x);
	if(s[0] == 0)
		return xt::zeros<float>(x.shape());
	return x/s;
}

xt::xarray<float> categroize(const xt::xarray<bool>& in)
{
	return categroize(TOKEN_TYPE_NUM, LEN_PER_TOKEN, in, false);
}

xt::xarray<int> loadDataset(std::string path)
{
	xt::xarray<int> dataset = xt::load_npy<int>(path);
	return dataset;
}

size_t sampleFromDistribution(const xt::xarray<float>& dist)
{
	//return xt::argmax(dist)[0];
	float v = xt::random::rand<float>({1})[0];
	float s = 0;
	size_t index = 0;
	for(auto p : dist) {
		s += p;
		if(v <= s)
			return index;
		index++;
	}
	return 0;
}


struct Model
{
	Model(): tm({TOKEN_TYPE_NUM*LEN_PER_TOKEN}, TP_DEPTH, 255, 8192)
	{
		tm->setMinThreshold(LEN_PER_TOKEN*0.35f+1);
		tm->setActivationThreshold(LEN_PER_TOKEN*0.25f);
		tm->setMaxNewSynapseCount(128);
		tm->setPermanenceIncrement(0.030);
		tm->setPermanenceDecrement(0.032);
		tm->setConnectedPermanence(0.24);
		tm->setPredictedSegmentDecrement((1.f/TOKEN_TYPE_NUM)*tm->getPermanenceIncrement()*1.7f);
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

	size_t predictNextToken(size_t currentToken)
	{
		auto res = predict(encode(currentToken));
		auto prop = xt::eval(softmax(categroize(res)));

		if(xt::sum(prop)[0] == 0)
			return -1;
		return sampleFromDistribution(prop);
	}

	std::vector<size_t> continousPredict(size_t init_token, int num = -1)
	{
		std::vector<size_t> res;
		size_t token = init_token;
		for(int i=0;i<num or i<0; i++) {
			token = predictNextToken(token);
			res.push_back(token);

			if(token == -1)
				break;
		}
		return res;
	}

	void reset()
	{
		tm.reset();
	}

	TemporalMemory tm;
};

inline void ptint(std::vector<size_t> tokens, bool init_upper = false)
{
	bool upper = init_upper;
	for(auto token : tokens) {
		if(token == -1) {
			std::cout << "<NO PREDICTION>\n";
			break;
		}

		std::cout << characterFromIndex(token, upper);
		if(token != 29 and upper == true)
			upper = false;
		if(token == 0)
			upper = true;
	}
}

xt::xarray<bool> noise(xt::xarray<float>::shape_type shape, float p = 0.01f)
{
	static std::mt19937 eng;
	return (xt::random::rand<float>(shape, 0,1, eng) < p) && (xt::random::rand<float>(shape, 0,1, eng) < 0.5);
}

int main()
{
	auto dataset = loadDataset("dataset.npy");
	Model model;

	std::cout << "traning temporal memory..." << std::endl;
	for(int i=0;i<20;i++) {
		for(auto token : dataset) {
			//Add some noise to break symmetry
			model.train(encode(token) ^ noise({TOKEN_TYPE_NUM*LEN_PER_TOKEN}, 0.0005));
			//if (token == 0)
			//	model.reset();
		}
		model.reset();
		std::cout << i << "\r" << std::flush;
	}
	std::cout << "\n";



	std::cout << "Genetare random text....\n";
	auto tokens = model.continousPredict(29, 400); //29 = ' '
	ptint(tokens, true);
	model.reset();



	std::cout << "\n\nFinishing a sentence....\n";
	auto test = loadDataset("test.npy");
	xt::xarray<bool> res;
	for(auto token : test) {
		std::cout << characterFromIndex(token);
		res = model.predict(encode(token));
	}

	int token = sampleFromDistribution(softmax(categroize(res)));
	std::cout << characterFromIndex(token);

	tokens = model.continousPredict(token, 40);
	ptint(tokens);
	model.reset();
	std::cout << "\n";



	for(int token=0;token<TOKEN_TYPE_NUM;token++) {
		std::cout << "\n\nGenerate random text started by " << characterFromIndex(token) << ":\n";
		std::cout << characterFromIndex(token, true);
		auto tokens = model.continousPredict(token, 40);
		ptint(tokens);
		model.reset();
	}
	
	std::cout << std::endl;
}
