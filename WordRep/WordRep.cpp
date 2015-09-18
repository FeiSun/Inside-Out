#include "WordRep.h"

WordRep::WordRep(int iter, int window, int min_count, int table_size, int word_dim, int negative,
	float subsample_threshold, float init_alpha, float min_alpha, int num_threads, string model, bool binary):
iter(iter),  window(window), min_count(min_count), table_size(table_size), word_dim(word_dim), 
	negative(negative), subsample_threshold(subsample_threshold), init_alpha(init_alpha),
	min_alpha(min_alpha), num_threads(num_threads), model(model), binary(binary),
	generator(rd()), distribution_table(0, table_size - 1),
	uni_dis(0.0, 1.0), distribution_window(0, window < 1 ? 0 : window - 1)
{
	doc_num = 0;
	total_words = 0;
	ep = numeric_limits<float>::epsilon();
} 

WordRep::~WordRep(void)
{
}

inline bool comp(Word *w1, Word *w2)
{
	return w1->count > w2->count;
}

vector<string> &split(const std::string &s, char delim, vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<vector<string>> WordRep::line_docs(string filename)
{
	vector<vector<string>> docs;
	ifstream in(filename);
	string s;

	while (std::getline(in, s))
	{
		istringstream iss(s);
		docs.emplace_back(istream_iterator<string>{iss}, istream_iterator<string>{});
	}
	return std::move(docs);
}

void WordRep::make_table(vector<size_t>& table, vector<Word *>& vocab)
{
	table.resize(table_size);
	size_t vocab_size = vocab.size();
	float power = 0.75f;
	float train_words_pow = 0.0f;

	vector<float> word_range(vocab.size());
	for(size_t i = 0; i != vocab_size; ++i)
	{
		word_range[i] = pow((float)vocab[i]->count, power);
		train_words_pow += word_range[i];
	}

	size_t idx = 0;
	float d1 = word_range[idx] / train_words_pow;
	float scope = table_size * d1;
	for(int i = 0; i < table_size; ++i)
	{
		table[i] = vocab[idx]->index;
		if(i > scope && idx < vocab_size - 1)
		{
			d1 += word_range[++idx] / train_words_pow;
			scope = table_size * d1;
		}
		else if(idx == vocab_size - 1)
		{
			for(; i < table_size; ++i)
				table[i] = vocab[idx]->index;
			break;
		}
	}
}

void WordRep::precalc_sampling()
{
	size_t vocab_size = W_vocab.size();
	size_t word_count = 0;

	float threshold_count  = subsample_threshold * total_words;

	if(subsample_threshold > 0)
		for(auto& w: W_vocab)
			w->sample_probability = std::min(float((sqrt(w->count / threshold_count) + 1) * threshold_count / w->count), (float)1.0);
	else
		for(auto& w: W_vocab)
			w->sample_probability = 1.0;
}

void WordRep::segment_vocab(string filename)
{
	ifstream in(filename);
	string s, w;
	int i = M_vocab.size();

	while (std::getline(in, s))
	{
		istringstream iss(s);
		iss >> w;

		auto it = W_vocab_hash.find(w);
		if (it == W_vocab_hash.end()) continue;
		Word *word = it->second.get();

		while (iss >> w)
		{
			if(M_vocab_hash.count(w) == 0)
			{
				Word *m = new Word(i++, word->count, w);
				M_vocab.push_back(m);
				M_vocab_hash[w] = WordP(m);
				word_morpheme[word].push_back(m);
			}
			else
			{
				Word* m = M_vocab_hash[w].get();
				m->count += word->count;
				word_morpheme[word].push_back(m);
			}
		}
	}
}

void WordRep::build_vocab(string filename, string mor_file)
{
	ifstream in(filename);
	string s, w;
	unordered_map<string, size_t> word_cn;

	while (std::getline(in, s))
	{
		doc_num++;
		istringstream iss(s);
		while (iss >> w)
		{
			if(word_cn.count(w) > 0)
				word_cn[w]++;
			else
				word_cn[w] = 1;
		}
	}
	in.close();
	//ignore words apper less than min_count
	for(auto kv: word_cn)
	{
		if(kv.second < min_count)
			continue;

		Word *w = new Word(0, kv.second,  kv.first);
		W_vocab.push_back(w);
		W_vocab_hash[w->text] = WordP(w);
		total_words += kv.second;
	}
	//update word index
	size_t vocab_size = W_vocab.size();
	sort(W_vocab.begin(), W_vocab.end(), comp);
	for(size_t i = 0; i < vocab_size; ++i)
	{
		W_vocab[i]->index = i;
	}
	
	make_table(this->W_table, this->W_vocab);
	
	precalc_sampling();

	if(model == "seing" || model == "boeing")
	{
		segment_vocab(mor_file);
		make_table(this->M_table, this->M_vocab);
	}
}

void WordRep::init_weights()
{
	std::uniform_real_distribution<float> distribution(-0.5, 0.5);
	auto uniform = [&] (int) {return distribution(generator);};

	W = RMatrixXf::NullaryExpr(W_vocab.size(), word_dim, uniform) / (float)word_dim;
	C = RMatrixXf::NullaryExpr(W_vocab.size(), word_dim, uniform) / (float)word_dim; //RMatrixXf::Zero(vocab_size - 1, word_dim);
	//if M_vocab exist
	M = RMatrixXf::NullaryExpr(M_vocab.size(), word_dim, uniform) / (float)word_dim;
}

vector<vector<Word *>> WordRep::build_docs(string filename)
{
	ifstream in(filename);
	string s, w;

	vector<vector<Word *>> docs;

	while (std::getline(in, s))
	{
		vector<Word *> doc;
		istringstream iss(s);

		while (iss >> w)
		{
			auto it = W_vocab_hash.find(w);
			if (it == W_vocab_hash.end()) continue;
			Word *word = it->second.get();

			doc.push_back(word);
		}
		docs.push_back(std::move(doc));
	}
	in.close();

	return std::move(docs);
}


void WordRep::negative_sampling(float alpha, Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad, 
	                            RMatrixXf& target_matrix, vector<size_t>& table)
{
	unordered_map<size_t, uint8_t> targets;
	for (int i = 0; i < negative; ++i)
		targets[table[distribution_table(generator)]] = 0;

	targets[predict_word->index] = 1;

	for (auto it: targets)
	{
		float f = target_matrix.row(it.first).dot(project_rep);
		f = 1.0 / (1.0 + exp(-f));
		float g = it.second - f;

		project_grad += g * target_matrix.row(it.first);
		RowVectorXf l2_grad = g * project_rep;

		target_matrix.row(it.first) += alpha * l2_grad;
	}
}


void WordRep::train_seing(vector<vector<Word *>>& docs)
{
	vector<long> sample_idx(docs.size());
	std::iota(sample_idx.begin(), sample_idx.end(), 0);

	long long wn = 0;
	float alpha = init_alpha;

	for(int it = 0; it < iter; ++it)
	{
		cout << "iter:" << it <<endl;
		std::shuffle(sample_idx.begin(), sample_idx.end(), generator);

        #pragma omp parallel for
		for(int i = 0; i < doc_num; ++i)
		{
			if(i % 10 == 0)
			{
				alpha = std::max(min_alpha, float(init_alpha * (1.0 - 1.0 / iter * wn / total_words)));
				#ifdef DEBUG
				printf("\ralpha: %f  Progress: %f%%", alpha, 100.0 / iter * wn / total_words);
				std::fflush(stdout);
				#endif
			}

			long doc_id = sample_idx[i];
			vector<Word *>& doc = docs[doc_id];
			int doc_len = doc.size();
			RowVectorXf neu1_grad = RowVectorXf::Zero(word_dim);

			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				if(current_word->sample_probability < uni_dis(generator))
					continue;

				int reduced_window = distribution_window(generator);
				int index_begin = max(0, j - window + reduced_window);
				int index_end = min((int)doc_len, j + window + 1 - reduced_window);
				
				//outer
				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;
					if(doc[m]->sample_probability < uni_dis(generator))
						continue;

					neu1_grad.setZero();

					RowVectorXf neu1 = W.row(current_word->index);
					negative_sampling(alpha, doc[m], neu1, neu1_grad, C, W_table);

					W.row(current_word->index) += alpha * neu1_grad;
				}

				//inner
				if(word_morpheme.count(current_word) == 0)
					continue;

				for(auto mor: word_morpheme[current_word])
				{
					neu1_grad.setZero();
					RowVectorXf neu1 = W.row(current_word->index);
					negative_sampling(alpha, mor, neu1, neu1_grad, M, M_table);

					W.row(current_word->index) += alpha * neu1_grad;
				}
				
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void WordRep::train_boeing(vector<vector<Word *>>& docs)
{
	vector<int> sample_idx(docs.size());
	std::iota(sample_idx.begin(), sample_idx.end(), 0);

	long long wn = 0;
	float alpha = init_alpha;

	for(int it = 0; it < iter; ++it)
	{
		cout << "iter:" << it <<endl;
		std::shuffle(sample_idx.begin(), sample_idx.end(), generator);

        #pragma omp parallel for
		for(int i = 0; i < docs.size(); ++i)
		{
			if(i % 10 == 0)
			{
				alpha = std::max(min_alpha, float(init_alpha * (1.0 - 1.0 / iter * wn / total_words)));
				#ifdef DEBUG
				printf("\ralpha: %f  Progress: %f%%", alpha, 100.0 / iter * wn / total_words);
				std::fflush(stdout);
				#endif
			}

			long doc_id = sample_idx[i];
			auto doc = docs[doc_id];
			size_t doc_len =  doc.size();

			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				if(current_word->sample_probability < uni_dis(generator))
					continue;

				int reduced_window = distribution_window(generator);
				int index_begin = max(0, j - window + reduced_window);
				int index_end = min((int)doc_len, j + window + 1 - reduced_window);

				RowVectorXf neu1 = RowVectorXf::Zero(word_dim);
				RowVectorXf neu1_grad = RowVectorXf::Zero(word_dim);

				vector<size_t> c_idx; 
				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;
					if(doc[m]->sample_probability < uni_dis(generator))
						continue;

					neu1 += C.row(doc[m]->index);
					c_idx.push_back(doc[m]->index);
				}

				if(c_idx.size() > 0)
				{
					neu1 /= c_idx.size();

					negative_sampling(alpha, current_word, neu1, neu1_grad, W, W_table);

					neu1_grad /= c_idx.size();

					for(auto m: c_idx)
					{
						C.row(m) += alpha * neu1_grad;
					}
				}

				//inner
				neu1.setZero();
				neu1_grad.setZero();

				if(word_morpheme.count(current_word) == 0)
					continue;

				for(auto mor: word_morpheme[current_word])
				{
					neu1 += M.row(mor->index);
				}

				neu1 /= word_morpheme[current_word].size();
				negative_sampling(alpha, current_word, neu1, neu1_grad, W, W_table);
				neu1_grad /= word_morpheme[current_word].size();

				for(auto mor: word_morpheme[current_word])
				{
					M.row(mor->index) += alpha * neu1_grad;;
				}
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}


void WordRep::train(string filename, string mor_file)
{
	build_vocab(filename, mor_file);
	init_weights();
	vector<vector<Word *>> docs = build_docs(filename);

	if(model == "seing")
		train_seing(docs);
	else if(model == "boeing")
		train_boeing(docs);
}

void WordRep::save_vocab(string vocab_filename)
{
	ofstream out(vocab_filename, std::ofstream::out);
	for(auto& v: W_vocab)
		out << v->index << " " << v->count << " " << v->text << endl;
	out.close();
}

void WordRep::save_vec(string filename, const RMatrixXf& data, vector<Word *>& vocab, bool binary)
{
	IOFormat CommaInitFmt(StreamPrecision, DontAlignCols);

	if(binary)
	{
		std::ofstream out(filename, std::ios::binary);
		char blank = ' ';
		char enter = '\n'; 
		int size = sizeof(char);
		int r_size = data.cols() * sizeof(RMatrixXf::Scalar);

		RMatrixXf::Index r = data.rows();
		RMatrixXf::Index c = data.cols();
		out.write((char*) &r, sizeof(RMatrixXf::Index));
		out.write(&blank, size);
		out.write((char*) &c, sizeof(RMatrixXf::Index));
		out.write(&enter, size);

		for(auto v: vocab)
		{
			out.write(v->text.c_str(), v->text.size());
			out.write(&blank, size);
			out.write((char*) data.row(v->index).data(), r_size);
			out.write(&enter, size);
		}
		out.close();
	}
	else
	{
		ofstream out(filename);

		out << data.rows() << " " << data.cols() << std::endl;

		for(auto v: vocab)
		{
			out << v->text << " " << data.row(v->index).format(CommaInitFmt) << endl;
		}
		out.close();
	}
}

