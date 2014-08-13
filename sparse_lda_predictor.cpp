#include <tr1/unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <functional>
#include <ext/functional>
using namespace std;
using namespace __gnu_cxx;
using tr1::unordered_map;


namespace lda {

ostream& operator<<(ostream& out, vector<int>& v)
{
    for (size_t i=0; i<v.size(); ++i)
        out<<v[i]<<" ";
    return out;
}

ostream& operator<<(ostream& out, vector<float>& v)
{
    for (size_t i=0; i<v.size(); ++i)
        out<<v[i]<<" ";
    return out;
}


class LdaModel {
public:
    static inline LdaModel& get_instance(string model_file, int num_topic, float alpha)
    {
        static LdaModel lda_model(model_file, num_topic, alpha);
        //if (NULL == _p_lda_model)
        //{
        //    _p_lda_model = new LdaModel(model_file, num_topic, alpha);
        //}
        return lda_model;
    }

    ~LdaModel()
    {
        for (unordered_map<int, unordered_map<int, float>* >::iterator iter=_wor2top.begin();
             iter != _wor2top.end();
             ++iter)
        {    delete iter->second;  }
    }

private:
    LdaModel(const string& model_file, int num_topic, float alpha) 
     : _num_topic(num_topic), _alpha(alpha)
    {
        load_model(model_file, num_topic);
    }

    // load model file
    // format:  topic_id  \t  wordid:count space word:count ...
    void load_model(const string& model_file, int num_topic)
    {
        _num_topic = num_topic;
        _topsum.resize(_num_topic, 0.0f);

        ifstream ifs(model_file.c_str());
        string buf;
        while(getline(ifs, buf))
        {
            istringstream ss(buf);
            int topic_id;
            ss >> topic_id;

            string item;
            while(ss >> item)  
            {
                vector<string> tokens;
                boost::split(tokens, item, boost::is_any_of(":"));
                int wordid = boost::lexical_cast<int>(tokens[0]);
                float count = boost::lexical_cast<float>(tokens[1]);
                _topsum[topic_id] += count;

                if (_wor2top.find(wordid) == _wor2top.end())
                {
                    _wor2top[wordid] = new unordered_map<int, float>(); 
                }
                (*_wor2top[wordid])[topic_id] = count;
            }
        }
    }
   
    // disallow copy and assignment
    LdaModel(const LdaModel&);
    LdaModel& operator = (const LdaModel&);

public:
    // wordid  topicid::count
    unordered_map<int, unordered_map<int, float>* > _wor2top;
    // topic's total word count 
    vector<float> _topsum; 
    int _num_topic;
    float _alpha;

};

//LdaModel* LdaModel::_p_lda_model = NULL;


class SparseLdaPredictor{
public:
    SparseLdaPredictor(LdaModel& lda_model) : _lda_model(lda_model) { }

    friend class LdaModel;
    
    void predict(const vector<int>& word_vector, int max_step, vector<pair<int, int> >& topic_vector)
    {
       for (size_t i=0; i<word_vector.size(); ++i)
       {
           if (_lda_model._wor2top.find(word_vector[i]) != _lda_model._wor2top.end())
               _doc.push_back(word_vector[i]);
       }
       _len = _doc.size();

       init_predictor();
       // start rt lda inference

       int step = 0;
       while (step < max_step)
       {
           for (int i=0; i<_len; i++)
           {
               int old_topic = _wor2top[i];
               int word = _doc[i];
               cout<<"-------- step "<<step<<", word "<<word<<", old_topic "<<old_topic<<"-----------"<<endl;
               cout<<_wor2top<<endl;
               unordered_map<int, float> *p_top_of_wor = _lda_model._wor2top[word];
               unordered_map<int, float> prob;
               for (unordered_map<int, float>::iterator iter = p_top_of_wor->begin();
                    iter != p_top_of_wor->end();
                    ++iter)
               {
                   int cur_topic = iter->first;
                   int adjust = 0;
                   if (old_topic == cur_topic && 
                       _doc2top.find(cur_topic) != _doc2top.end() && 
                       _doc2top[cur_topic] > 0)
                   {
                       adjust = 1;
                   }
                   int theta = _doc2top[cur_topic] - adjust;
                   prob[cur_topic] = iter->second / _lda_model._topsum[cur_topic]
                                       * (theta + _lda_model._alpha);
                   cout<<"cur_topic="<<cur_topic
                       <<" theta="<<theta
                       <<" p(w|z)="<<iter->second / _lda_model._topsum[iter->first]
                       <<" p(w z)="<<prob[cur_topic]<<endl;
               }
               int sample = random_sparse_multinomial(prob);

               // adjust topic assignment
               if (old_topic != sample)
               {
                   _wor2top[i] = sample;
                   _doc2top[old_topic] --;
                   
                   if (_doc2top.find(sample) != _doc2top.end())
                   {
                       _doc2top[sample] ++;
                   }
                   else
                   {
                       _doc2top[sample] = 1;
                   }
               }
               cout<<"+++++ sample="<<sample<<"  after adjust: "<<_wor2top<<endl;
           }//end for
           step++;
       }// end while
      
       topic_vector.resize(_doc.size());
       transform(_doc.begin(), _doc.end(), _wor2top.begin(), topic_vector.begin(), make_pair<int, int>);
    }


private:
    void init_predictor()
    {
        _wor2top.resize(_len); 
        for (int i = 0; i<_len; ++i)
        {
            int topic = random_topic();
            _wor2top[i] = topic;
            if (_doc2top.find(topic) == _doc2top.end())
                _doc2top[topic] = 1;
            else
                _doc2top[topic] += 1;
        }
    }


    inline int random_topic()
    {
        return static_cast<int>(rand() / static_cast<double>(RAND_MAX) * _lda_model._num_topic);
    }

    inline int random_sparse_multinomial(unordered_map<int, float>& prob)
    {
        vector<pair<int, float> > prob_vec(prob.begin(), prob.end());
        size_t sz = prob.size();
        for (size_t i=1; i<sz; ++i)
            prob_vec[i].second += prob_vec[i-1].second;
        double rdm = rand() / static_cast<double>(RAND_MAX) * prob_vec[sz-1].second;
        vector<pair<int, float> >::iterator iter = find_if(prob_vec.begin(), prob_vec.end(), 
                                  compose1(bind1st(less_equal<double>(), rdm), _Select2nd<pair<int, float> >()));
        return iter->first;
    }

private:
    LdaModel&  _lda_model;
    // word vector
    vector<int> _doc;
    // topic vector for each word
    vector<int> _wor2top;
    // doc length
    int _len;
    // topic -> count
    unordered_map<int, int> _doc2top;
};

}


using namespace lda;

#define rdtsc(low,high) __asm__ \
 __volatile__("rdtsc" : "=a" (low), "=d" (high))
unsigned long long get_cycles()
{
    unsigned low, high;
    unsigned long long val;
    rdtsc(low,high);
    val = high;
    val = (val << 32) | low; //将 low 和 high 合成一个 64 位值
    return val;
}
double mhz = 2128.054 * 1000;


int main(int argc, char *argv[])
{
    
    if (argc < 5)
    {
        cout<<"Usage : "<< argv[0]<<" alpha num_topic model_file query"<<endl;
        return 0;
    }
    srand(time(NULL));
    float alpha = boost::lexical_cast<float>(argv[1]);
    int num_topic = boost::lexical_cast<int>(argv[2]);
    string model_file = argv[3];
    string query = argv[4];

    int max_step = 10;

    long long t_start, t_end;

    LdaModel& lda_model = LdaModel::get_instance(model_file, num_topic, alpha);

    SparseLdaPredictor predictor(lda_model);

    vector<int> input;
    istringstream iss(query);
    string buf;
    while (iss>>buf) { input.push_back(boost::lexical_cast<int>(buf)); }
    
    t_start = get_cycles();
    vector<pair<int, int> > output;
    predictor.predict(input, max_step, output);
    t_end = get_cycles();

    cout<<"%%%%%%%%%%% final result %%%%%%%%%%%%%%%"<<endl;
    for(size_t i=0; i<output.size(); ++i)
        cout<<output[i].first<<" "<<output[i].second<<endl;
    cout<<"cost "<<(t_end - t_start)/mhz<<" ms"<<endl;
    return 0;
}
