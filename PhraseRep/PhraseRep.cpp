#include "PhraseRep.h"

PhraseRep::PhraseRep(int iter, int window, int min_count, int table_size, int word_dim, int negative,
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

PhraseRep::~PhraseRep(void)
{
}

inline bool comp(Word *w1, Word *w2)
{
	return w1->count > w2->count;
}

vector<std::string> &split(const std::string &s, char delim, vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<vector<string>> PhraseRep::line_docs(string filename)
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

void PhraseRep::make_table(vector<size_t>& table, vector<Word *>& vocab)
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

void PhraseRep::precalc_sampling()
{
	size_t vocab_size = P_vocab.size();
	size_t word_count = 0;

	float threshold_count  = subsample_threshold * total_words;

	if(subsample_threshold > 0)
		for(auto& w: P_vocab)
			w->sample_probability = std::min(float((sqrt(w->count / threshold_count) + 1) * threshold_count / w->count), (float)1.0);
	else
		for(auto& w: P_vocab)
			w->sample_probability = 1.0;
}

void PhraseRep::build_vocab(string filename)
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
		P_vocab.push_back(w);
		P_vocab_hash[w->text] = WordP(w);
		total_words += kv.second;
	}
	//update word index
	size_t vocab_size = P_vocab.size();
	sort(P_vocab.begin(), P_vocab.end(), comp);
	for(size_t i = 0; i < vocab_size; ++i)
	{
		P_vocab[i]->index = i;
	}

	make_table(this->P_table, this->P_vocab);
	precalc_sampling();

	for(auto p: P_vocab)
	{
		Word *w = new Word(p->index, p->count,  p->text);
		C_vocab.push_back(w);
		C_vocab_hash[p->text] = WordP(w);
	}

	if(model == "seing" || model == "boeing" || model == "cwecbow")
		segment_vocab();

	make_table(this->C_table, this->C_vocab);
}

void PhraseRep::segment_vocab()
{
	int i = C_vocab.size();

	for(auto kv: P_vocab_hash)
	{
		string w = kv.first;
		Word *phrase = kv.second.get();

		if(w.find('_') != string::npos)
		{
			vector<string> elems;
			split(w, '_', elems);

			for(auto e: elems)
			{
				if(C_vocab_hash.count(e) == 0)
				{
					Word *c = new Word(i++, 1, e);
					C_vocab.push_back(c);
					C_vocab_hash[e] = WordP(c);

					phrase_word[phrase].push_back(C_vocab_hash[e].get());
				}
				else
				{
					phrase_word[phrase].push_back(C_vocab_hash[e].get());
				}
			}
		}
	}
}

void PhraseRep::init_weights()
{
	std::uniform_real_distribution<float> distribution(-0.5, 0.5);
	auto uniform = [&] (int) {return distribution(generator);};

	P = RMatrixXf::NullaryExpr(P_vocab.size(), word_dim, uniform) / (float)word_dim;
	C = RMatrixXf::NullaryExpr(C_vocab.size(), word_dim, uniform) / (float)word_dim;
}

vector<vector<Word *>> PhraseRep::build_docs(string filename)
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
			auto it = P_vocab_hash.find(w);
			if (it == P_vocab_hash.end()) continue;
			Word *word = it->second.get();

			doc.push_back(word);
		}
		docs.push_back(std::move(doc));
	}
	in.close();

	return std::move(docs);
}

void PhraseRep::negative_sampling(float alpha, Word* predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad, 
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

void PhraseRep::train_sg(vector<vector<Word *>>& docs)
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

			vector<Word *>& doc = docs[sample_idx[i]];
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

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;

					RowVectorXf neu1 = P.row(current_word->index);
					neu1_grad.setZero();
					negative_sampling(alpha, doc[m], neu1, neu1_grad, C, C_table);

					P.row(current_word->index) += alpha * neu1_grad;
				}
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void PhraseRep::train_cbow(vector<vector<Word *>>& docs)
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

			auto doc = docs[sample_idx[i]];
			size_t doc_len =  doc.size();

			for(int j = 0; j < doc_len; ++j)
			{
				Word* current_word = doc[j];
				if(current_word->sample_probability < uni_dis(generator))
					continue;

				int reduced_window = distribution_window(generator);
				int index_begin = max(0, j - window + reduced_window);
				int index_end = min((int)doc_len, j + window + 1 - reduced_window);
				if (index_end <= (index_begin + 1)) continue;

				RowVectorXf neu1 = RowVectorXf::Zero(word_dim);
				RowVectorXf neu1_grad = RowVectorXf::Zero(word_dim);

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;
					neu1 += C.row(doc[m]->index);
				}
				if(index_end - index_begin > 1)
					neu1 /= index_end - index_begin - 1;

				negative_sampling(alpha, current_word, neu1, neu1_grad, P, P_table);

				if(index_end - index_begin > 1)
					neu1_grad /= index_end - index_begin - 1;

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;

					C.row(doc[m]->index) += alpha * neu1_grad;
				}
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void PhraseRep::train_seing(vector<vector<Word *>>& docs)
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

					neu1_grad.setZero();

					RowVectorXf neu1 = P.row(current_word->index);
					negative_sampling(alpha, doc[m], neu1, neu1_grad, C, C_table);

					P.row(current_word->index) += alpha * neu1_grad;
				}

				//inner
				if(phrase_word.count(current_word) == 0)
					continue;

				for(auto c: phrase_word[current_word])
				{
					neu1_grad.setZero();
					RowVectorXf neu1 = P.row(current_word->index);
					negative_sampling(alpha, c, neu1, neu1_grad, C, C_table);

					P.row(current_word->index) += alpha * neu1_grad;
				}
			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}

void PhraseRep::train_boeing(vector<vector<Word *>>& docs)
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

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;
					neu1 += C.row(doc[m]->index);
				}
				if(index_end - index_begin > 1)
					neu1 /= index_end - index_begin - 1;

				negative_sampling(alpha, current_word, neu1, neu1_grad, P, P_table);

				if(index_end - index_begin > 1)
					neu1_grad /= index_end - index_begin - 1;

				for(int m = index_begin; m < index_end; ++m)
				{
					if(m == j) continue;

					C.row(doc[m]->index) += alpha * neu1_grad;
				}

				//inner
				neu1.setZero();
				neu1_grad.setZero();

				if(phrase_word.count(current_word) == 0)
					continue;

				for(auto c: phrase_word[current_word])
				{
					neu1 += C.row(c->index);
				}
				neu1 /= phrase_word[current_word].size();

				negative_sampling(alpha, current_word, neu1, neu1_grad, P, P_table);

				neu1_grad /= phrase_word[current_word].size();

				for(auto c: phrase_word[current_word])
				{
					C.row(c->index) += alpha * neu1_grad;;
				}

			}

            #pragma omp atomic
			wn += doc_len;
		}
	}
}



void PhraseRep::train(string filename)
{
	build_vocab(filename);
	init_weights();
	vector<vector<Word *>> docs = build_docs(filename);

	if(model == "sg")
		train_sg(docs);
	else if(model == "cbow")
		train_cbow(docs);
	else if(model == "seing")
		train_seing(docs);
	else if(model == "boeing")
		train_boeing(docs);
}

void PhraseRep::save_vocab(vector<Word *>& vocab, string vocab_filename)
{
	ofstream out(vocab_filename, std::ofstream::out);
	for(auto& v: vocab)
		out << v->index << " " << v->count << " " << v->text << endl;
	out.close();
}

void PhraseRep::save_vec(string filename, const RMatrixXf& data, vector<Word *>& vocab, bool binary)
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
