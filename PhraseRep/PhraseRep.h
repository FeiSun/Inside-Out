#pragma once
#include <vector>
#include <list>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <limits>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include "Word.h"

using namespace std;
using namespace Eigen;

#define EIGEN_NO_DEBUG
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;


class PhraseRep
{
public:
	int iter;
	int window;
	int min_count;
	int table_size;
	int word_dim;
	int negative;
	float subsample_threshold;
	float init_alpha;
	float min_alpha;
	float ep;
	int num_threads;
	long doc_num;
	long long total_words;

	bool binary;

	string model;

	vector<Word *> P_vocab;
	unordered_map<string, WordP> P_vocab_hash;
	vector<size_t> P_table;

	vector<Word *> C_vocab;
	unordered_map<string, WordP> C_vocab_hash;
	vector<size_t> C_table;

	unordered_map<Word *, vector<Word *>> phrase_word;

	std::uniform_int_distribution<int> distribution_table;
	std::uniform_real_distribution<float> uni_dis;
	std::uniform_int_distribution<int> distribution_window;

	RMatrixXf C;
	RMatrixXf P;

	std::random_device rd;
	std::mt19937 generator;

public:
	PhraseRep(int iter=5, int window=10, int min_count=5, int table_size=100000000, int word_dim=400,
		int negative=0, float subsample_threshold=0.0001,float init_alpha=0.025,
		float min_alpha=1e-6, int num_threads=1, string model="seing", bool binary=false);
	~PhraseRep(void);

	vector<vector<string>> line_docs(string filename);
	void reduce_vocab();
	void make_table(vector<size_t>& table, vector<Word *>& vocab);
	void precalc_sampling();
	void build_vocab(string filename);
	void segment_vocab();
	void init_weights();
	vector<vector<Word *>> build_docs(string filename);

	void negative_sampling(float alpha, Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad,
		                   RMatrixXf& target_matrix,  vector<size_t>& table);

	void train_sg(vector<vector<Word *>>& docs);
	void train_cbow(vector<vector<Word *>>& docs);

	void train_seing(vector<vector<Word *>>& docs);
	void train_boeing(vector<vector<Word *>>& docs);

	void train(string filename);

	void save_vocab(vector<Word *>& vocab, string vocab_filename);
	void save_vec(string filename, const RMatrixXf& data, vector<Word *>& vocab, bool binary=false);
};

