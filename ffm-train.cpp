#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "ffm.h"

#if defined USEOMP
#include <omp.h>
#endif

using namespace std;
using namespace ffm;

string train_help() {
    return string(
"usage: ffm-train [options] training_set_file [model_file]\n"
"\n"
"options:\n"
"-l <lambda>: set regularization parameter (default 0.00002)\n"
"-k <factor>: set number of latent factors (default 4)\n"
"-t <iteration>: set number of iterations (default 15)\n"
"-r <eta>: set learning rate (default 0.2)\n"
"-s <nr_threads>: set number of threads (default 1)\n"
"-p <path>: set path to the validation set\n"
"--quiet: quiet mode (no output)\n"
"--no-norm: disable instance-wise normalization\n"
"--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n");
}

struct Option {
    string tr_path;
    string va_path;
    string model_path;
    ffm_parameter param;
    bool quiet = false;
    ffm_int nr_threads = 1;
};

string basename(string path) {
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

Option parse_option(int argc, char **argv) {
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option opt;
    for(int x=0;x<128;x++){
        opt.param.multiplier[x] =1;
        opt.param.multiplier2[x] =1;
    }

    ffm_int i = 1;
    for(; i < argc; i++) {
        if(args[i].compare("-t") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify number of iterations after -t");
            i++;
            opt.param.nr_iters = atoi(args[i].c_str());
            if(opt.param.nr_iters <= 0)
                throw invalid_argument("number of iterations should be greater than zero");
        } else if(args[i].compare("-k") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify number of factors after -k");
            i++;
            opt.param.k = atoi(args[i].c_str());
            if(opt.param.k <= 0)
                throw invalid_argument("number of factors should be greater than zero");
        } else if(args[i].compare("-r") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify eta after -r");
            i++;
            opt.param.eta = atof(args[i].c_str());
            if(opt.param.eta <= 0)
                throw invalid_argument("learning rate should be greater than zero");
        } else if(args[i].compare("-l") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify lambda after -l");
            i++;
            opt.param.lambda = atof(args[i].c_str());
            if(opt.param.lambda < 0)
                throw invalid_argument("regularization cost should not be smaller than zero");
        } else if(args[i].compare("-s") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify number of threads after -s");
            i++;
            opt.nr_threads = atoi(args[i].c_str());
            if(opt.nr_threads <= 0)
                throw invalid_argument("number of threads should be greater than zero");
        } else if(args[i].compare("-p") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            opt.va_path = args[i];
        } else if(args[i].compare("--no-norm") == 0) {
            opt.param.normalization = false;
        } else if(args[i].compare("--quiet") == 0) {
            opt.quiet = true;
        } else if(args[i].compare("--auto-stop") == 0) {
            opt.param.auto_stop = true;
        } else if(args[i].compare("-b") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify beta after -b");
            i++;
            opt.param.beta = atof(args[i].c_str());
            if(opt.param.beta < 0)
                throw invalid_argument("beta should not be smaller than zero");
        } else if(args[i].compare("-m") == 0){
            if(i == argc-1)
                throw invalid_argument("need to specify multiplier for different field comma separated after -m");
            i++;
            int len = args[i].length();
            char value[len+1];
            strcpy(value, args[i].c_str());
            char* field0 = strtok(value,":");
            char* multiplier0 = strtok(nullptr,",");
            opt.param.multiplier[atoi(field0)] = atof(multiplier0);
            while (true) {
                char* field = strtok(nullptr,":");
                char* multiplier = strtok(nullptr,",");
                if(field == nullptr){
                    break;
                }
                opt.param.multiplier[atoi(field)] = atof(multiplier);
            }

        }else if(args[i].compare("-m2") == 0){
            if(i == argc-1)
                throw invalid_argument("need to specify multiplier for different field comma separated after -m2");
            i++;
            int len = args[i].length();
            char value[len+1];
            strcpy(value, args[i].c_str());
            char* field0 = strtok(value,":");
            char* multiplier0 = strtok(nullptr,",");
            opt.param.multiplier2[atoi(field0)] = atof(multiplier0);
            while (true) {
                char* field = strtok(nullptr,":");
                char* multiplier = strtok(nullptr,",");
                if(field == nullptr){
                    break;
                }
                opt.param.multiplier2[atoi(field)] = atof(multiplier);
            }

        }else if(args[i].compare("-tr1") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify tryouts for fields after -tr1");
            i++;
            int len = args[i].length();
            char value[len+1];
            strcpy(value, args[i].c_str());
            char* field0 = strtok(value,":");
            char* try0 = strtok(nullptr,",");
            opt.param.try_out[atoi(field0)] = atoi(try0);
            while (true) {
                char* field = strtok(nullptr,":");
                char* tryout = strtok(nullptr,",");
                if(field == nullptr){
                    break;
                }
                opt.param.try_out[atoi(field)] = atoi(tryout);
            }
        }else if(args[i].compare("-tr2") == 0) {
            if(i == argc-1)
                throw invalid_argument("need to specify tryouts for fields after -tr2");
            i++;
            int len = args[i].length();
            char value[len+1];
            strcpy(value, args[i].c_str());
            char* field0 = strtok(value,":");
            char* try0 = strtok(nullptr,",");
            opt.param.try_out_2[atoi(field0)] = atoi(try0);
            while (true) {
                char* field = strtok(nullptr,":");
                char* tryout = strtok(nullptr,",");
                if(field == nullptr){
                    break;
                }
                opt.param.try_out_2[atoi(field)] = atoi(tryout);
            }
        }
        else{
            break;
        }
    }

    if(i != argc-2 && i != argc-1)
        throw invalid_argument("cannot parse command\n");

    opt.tr_path = args[i];
    i++;

    if(i < argc) {
        opt.model_path = string(args[i]);
    } else if(i == argc) {
        opt.model_path = basename(opt.tr_path) + ".model";
    } else {
        throw invalid_argument("cannot parse argument");
    }

    return opt;
}
void save_best_params(ffm_float best_m1[128], ffm_float best_m2[128] , string path){
    std::ofstream f_out(path);
    std::string out;
    f_out << "best_m1 \n";
    for(int i=0;i<10;i++){
        f_out << best_m1[i]<<" ";
    }
    f_out << "\n best_m2 \n";
    for(int i=0;i<10;i++){
        f_out << best_m2[i]<<" ";
    }
    fout<<"\n";
    f_out.close();
}

int train_on_disk(Option opt) {
    string tr_bin_path = basename(opt.tr_path) + ".bin";
    string va_bin_path = opt.va_path.empty()? "" : basename(opt.va_path) + ".bin";

    ffm_read_problem_to_disk(opt.tr_path, tr_bin_path);
    if(!opt.va_path.empty())
        ffm_read_problem_to_disk(opt.va_path, va_bin_path);


    ffm_double min_val_loss = 1e9;
    ffm_model best_model = ffm_train_on_disk(tr_bin_path.c_str(), va_bin_path.c_str(), opt.param, &min_val_loss);
    ffm_float best_m1[128];
    ffm_float best_m2[128];
    for(int i=0;i<128;i++){
        best_m1[i] = opt.param.multiplier[i];
        best_m2[i] = opt.param.multiplier2[i];
    }

    for(int i=0;i<5;i++) {
        for(int j=0;j<5;j++) {
            for (int k = - opt.param.try_out[i]; k < opt.param.try_out[i]; k++) {
                for(int l= -opt.param.try_out_2[j]; l<opt.param.try_out_2[j]; l++) {
                    ffm_double current_loss = 1e9;
                    opt.param.multiplier[i] = opt.param.multiplier[i] * pow(2,k);
                    opt.param.multiplier2[j] = opt.param.multiplier2[j] * pow(2,l);
                    ffm_model model = ffm_train_on_disk(tr_bin_path.c_str(), va_bin_path.c_str(), opt.param, &current_loss);

                    if(current_loss < min_val_loss){
                        copy_model(best_model,model);
                        min_val_loss = current_loss;
                        best_m1[i] = opt.param.multiplier[i];
                        best_m2[j] = opt.param.multiplier2[j];
                    }
                    opt.param.multiplier[i] = opt.param.multiplier[i] * pow(2,-k);
                    opt.param.multiplier2[j] = opt.param.multiplier2[j] * pow(2,-l);

                }
            }
        }
    }
    cout<< min_val_loss<<" hi  there "<< endl;
    ffm_save_model(best_model, opt.model_path);
    ffm_save_txt(best_model, opt.model_path+".txt");
    save_best_params(best_m1,best_m2,opt.model_path+".best_param");

    return 0;
}

int main(int argc, char **argv) {
    Option opt;
    try {
        opt = parse_option(argc, argv);
    } catch(invalid_argument &e) {
        cout << e.what() << endl;
        return 1;
    }

    if(opt.quiet)
        cout.setstate(ios_base::badbit);

    if(opt.param.auto_stop && opt.va_path.empty()) {
        cout << "To use auto-stop, you need to assign a validation set" << endl;
        return 1;
    }

#if defined USEOMP
    cout <<"threads "<< opt.nr_threads <<endl;
    omp_set_num_threads(opt.nr_threads);
#endif

    train_on_disk(opt);

    return 0;
}



