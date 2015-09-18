//g++ main.cpp WordRep.cpp -od2v -I/usr/local/include/eigen/ -std=c++11 -O3 -march=native -funroll-loops -lm -fopenmp
#include "WordRep.h"

void help()
{

	cout << "Word vector estimation toolkit v0.6" << endl << endl;
	cout << "Options:" << endl;
	cout << "Parameters for training:" << endl;
	cout << "\t-train <file>" << endl;
	cout << "\t\tUse text data from <file> to train the model" << endl;
	cout << "\t-mor_file <file>" << endl;
	cout << "\t\tUse word segmentation data from <file> to train the model" << endl;
	cout << "\t-word_output <file>" << endl;
	cout << "\t\tUse <file> to save the resulting word vectors" << endl;
	cout << "\t-word_size <int>"<< endl;
	cout << "\t\tSet size of word vectors; default is 100"<< endl;
	cout << "\t-window <int>"<< endl;
	cout << "\t\tSet max skip length between words; default is 5"<< endl;
	cout << "\t-subsample <float>"<< endl;
	cout << "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data"<< endl;
	cout << "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)"<< endl;
	cout << "\t-negative <int>" << endl;
	cout << "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)" << endl;
	cout << "\t-threads <int>" << endl;
	cout << "\t\tUse <int> threads (default 12)" << endl;
	cout << "\t-iter <int>" << endl;
	cout << "\t\tRun more training iterations (default 5)" << endl;
	cout << "\t-min-count <int>" << endl;
	cout << "\t\tThis will discard words that appear less than <int> times; default is 5" << endl;
	cout << "\t-alpha <float>" << endl;
	cout << "\t\tSet the starting learning rate; default is 0.025 for HDC and 0.05 for PDC" << endl;
	cout << "\t-binary <int>" << endl;
	cout << "\t\tSave the resulting vectors in binary moded; default is 0 (off)" << endl;
	cout << "\t-save-W_vocab <file>" << endl;
	cout << "\t\tThe vocabulary will be saved to <file>" << endl;
	cout << "\t-model <string>" << endl;
	cout << "\t\tThe model; default is continuous bag of words model(cbow) (use sg for skip-gram model)" << endl;
	cout << "\nExamples:" << endl;
	cout << "./wordrep -train data.txt -mor_file vocab_seg.txt -word_output vec.txt -size 200 -window 10 -subsample 1e-4 -negative 10 -model boeing -binary 0 -iter 20" << endl;
}

int ArgPos(char *str, int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
		if (!strcmp(str, argv[i])) {
			if (i == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return i;
		}
		return -1;
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int i = 0;
	if (argc == 1)
	{
		help();
		return 0;
	}

	string input_file = "";
	string mor_file = "";
	string w_output_file = "";
	string save_vocab_file = "";
	string read_vocab_file = "";
	string model = "boeing";
	int table_size = 100000000;
	int word_dim = 100;
	float init_alpha = 0.025f;
	int window = 5;
	float subsample_threshold = 1e-4;
	float min_alpha = init_alpha * 0.0001;
	int negative = 10;
	int num_threads = 12;
	int iter = 20;
	int min_count = 1;
	bool binary = false;

	if ((i = ArgPos((char *)"-word_size", argc, argv)) > 0)
		word_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
		input_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-mor_file", argc, argv)) > 0)
		mor_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-save-W_vocab", argc, argv)) > 0)
		save_vocab_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-read-W_vocab", argc, argv)) > 0)
		read_vocab_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0)
		if (atoi(argv[i + 1]) == 1)
			binary = true;
	if ((i = ArgPos((char *)"-model", argc, argv)) > 0)
		model = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
		init_alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-word_output", argc, argv)) > 0)
		w_output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-subsample", argc, argv)) > 0)
		subsample_threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
		num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
		iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);

	WordRep w2v(iter, window, min_count, table_size, word_dim, negative, subsample_threshold,
		init_alpha, min_alpha, num_threads, model, binary);

	omp_set_num_threads(num_threads);

	w2v.train(input_file, mor_file);
	if(w_output_file != "")
		w2v.save_vec(w_output_file, w2v.W, w2v.W_vocab);

}
