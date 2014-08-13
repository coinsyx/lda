#include <tr1/unordered_map>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
using namespace std;
using tr1::unordered_map;


namespace lda {

ostream& operator<<(ostream& out, vector<int>& v)
{
    for (size_t i=0; i<v.size(); ++i)
        out<<v[i]<<" ";
    return out;
}


class LdaModel {
public:
    static inline LdaModel* get_instance(string model_file, int num_topic, float alpha)
    {
        if (NULL == _p_lda_model)
        {
            //string model_file = "";
            //int num_topic = 0;
            //float alpha = 1.0; 
            _p_lda_model = new LdaModel(model_file, num_topic, alpha);
        }
        return _p_lda_model;
    }

private:
    LdaModel(const string& model_file, int num_topic, float alpha) 
     : _num_topic(num_topic), _alpha(alpha)
    {
        load_model(model_file, num_topic);
        calc_r();
        print_model_info();
    }

    // load model file
    // format:  topic_id  \t  wordid:count space word:count ...
    bool load_model(const string& model_file, int num_topic)
    {
        _num_topic = num_topic;
        _top2wor.resize(_num_topic, NULL);
        _topsum.resize(_num_topic, 0.0f);

        ifstream ifs(model_file.c_str());
        string buf;
        while(getline(ifs, buf))
        {
            istringstream ss(buf);
            int topic_id;
            ss >> topic_id;
            _top2wor[topic_id] = new unordered_map<int, float>();

            string item;
            while(ss >> item)  
            {
                vector<string> tokens;
                boost::split(tokens, item, boost::is_any_of(":"));
                int wordid = boost::lexical_cast<int>(tokens[0]);
                float count = boost::lexical_cast<float>(tokens[1]);
                (*_top2wor[topic_id])[wordid] = count;
                _topsum[topic_id] += count;

                if (_wor2top.find(wordid) == _wor2top.end())
                {
                    _wor2top[wordid] = new unordered_map<int, float>(); 
                }
                (*_wor2top[wordid])[topic_id] = count;
            }
        }
    }
   
    void calc_r()
    {
        // find max p(w_i | z_k) for each word_i
        for(int i=0; i< _num_topic; ++i)
        {
            float sum_count = _topsum[i];
            unordered_map<int, float>* one_topic = _top2wor[i];
            for (unordered_map<int, float>::iterator iter = one_topic->begin();
                 iter != one_topic->end();
                 ++iter)
            {
                int wordid = iter->first;
                float p = iter->second / sum_count;
                if (_R.find(wordid) == _R.end())
                {
                    _R.insert(std::make_pair<int, pair<int, float> >(wordid, pair<int, float>(i, p)));
                }
                else if (_R[wordid].second < p)
                {
                    _R[wordid].first = i;
                    _R[wordid].second = p;
                }
            }
        }
        // multiply alpha
        for(unordered_map<int, pair<int, float> >::iterator iter = _R.begin();
            iter != _R.end();
            ++iter)
        {
            (iter->second).second *= _alpha;
        }
    }
   
    void print_model_info()
    {
        cout<<"num_topic="<<_num_topic<<endl;
        cout<<"------- R -------"<<endl;
        for (unordered_map<int, pair<int, float> >::iterator iter = _R.begin();
             iter != _R.end();
             ++iter)
        {
            cout<<"word:"<<iter->first<<" topic:"<<(iter->second).first<<"  R="<<(iter->second).second<<endl;
        }
    }
    
    // disallow copy and assignment
    LdaModel(const LdaModel&);
    LdaModel& operator = (const LdaModel&);

public:
    // topic_id   wordid:count
    vector<unordered_map<int, float>* > _top2wor;
    // wordid  topicid::count
    unordered_map<int, unordered_map<int, float>* > _wor2top;
    // topic's total word count 
    vector<float> _topsum; 
    // R vector, wordid  <topic_id, p(z|w)> see wangyi's paper
    unordered_map<int, pair<int, float> > _R;
        
    int _num_topic;
    float _alpha;

    static LdaModel* _p_lda_model;
};

LdaModel* LdaModel::_p_lda_model = NULL;

class RtLdaPredictor
{
public:
    RtLdaPredictor(LdaModel* p_lda_model) : _p_lda_model(p_lda_model) { }

    void predict(const vector<int>& word_vector, int max_step, vector<int>& topic_vector)
    {
       copy(word_vector.begin(), word_vector.end(), back_inserter<vector<int> >(_doc)); 
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

               int max_topic = 0;
               float max_phi = 0.0;

               // max_k p(w|z_k) * (theta_k + alpha)
               unordered_map<int, float> *p_top_of_wor = _p_lda_model->_wor2top[word];
               for (unordered_map<int, float>::iterator iter = p_top_of_wor->begin();
                    iter != p_top_of_wor->end();
                    ++iter)
               {
                   int cur_topic = iter->first;
                   // \theta_k = 0, do not need process
                   if (_doc2top.find(cur_topic) == _doc2top.end() || 0 == _doc2top[cur_topic] )
                   {
                       cout<<"cur_topic["<<cur_topic<<"] not in _doc2top,  continue..."<<endl;
                       continue;
                   }
                   int adjust = cur_topic == old_topic ? 1 : 0;
                   int theta = _doc2top[cur_topic] - adjust;
                   if (theta == 0)
                   {
                       cout<<"theta=0, continue...  "
                           <<" _doc2top["<<cur_topic<<"]="<<_doc2top[cur_topic]
                           <<" adjust="<<adjust<<endl;
                       continue;
                   }
                   float phi = iter->second / _p_lda_model->_topsum[cur_topic] * (theta + _p_lda_model->_alpha);
                   cout<<"cur_topic="<<cur_topic
                       <<" theta="<<theta
                       <<" phi="<<phi
                       <<" max_phi="<<max_phi
                       <<" _R[word]="<<_p_lda_model->_R[word].second
                       <<endl;
                   if (phi > max_phi)
                   {
                       max_phi = phi;
                       max_topic = cur_topic;
                   }
               }
              
               // max_k { R, above value } 
               if (_p_lda_model->_R[word].second > max_phi)
               {
                   max_phi = _p_lda_model->_R[word].second;
                   max_topic = _p_lda_model->_R[word].first;
               }
               cout<<"max_topic:"<<max_topic<<" max_phi="<<max_phi<<endl;
               // adjust topic assignment
               if (old_topic != max_topic)
               {
                   _wor2top[i] = max_topic;
                   _doc2top[old_topic]--;
                   if (_doc2top.find(max_topic) != _doc2top.end())
                   {
                       _doc2top[max_topic] ++;
                   }
                   else
                   {
                       _doc2top[max_topic] = 1;
                   }
               }
               cout<<"after adjust"<<endl;
               cout<<_wor2top<<endl;
               
           }// end for
           step++;
       }// end while
       
       copy(_wor2top.begin(), _wor2top.end(), back_inserter<vector<int> >(topic_vector)); 
    }

private:
    void init_predictor()
    {
        _wor2top.resize(_len); 
        for (int i = 0; i<_len; ++i)
        {
            int topic = random_topic();
            _wor2top[i] = topic;
            cout<<"init topic "<<topic<<endl;
            if (_doc2top.find(topic) == _doc2top.end())
            {
                _doc2top[topic] = 1;
            }
            else
            {
                _doc2top[topic] += 1;
            }
        }
        cout<<_wor2top<<endl;
    }


    inline int random_topic()
    {
        return static_cast<int>(rand() / static_cast<double>(RAND_MAX) * _p_lda_model->_num_topic);
    }

private:

    LdaModel* _p_lda_model;

    vector<int> _doc;
    vector<int> _wor2top;
    int _len;
    // topic -> count
    unordered_map<int, int> _doc2top;
};

}

using namespace lda;

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        cout<<"Usage : "<< argv[0]<<" alpha num_topic model_file query"<<endl;
        return 0;
    }

    float alpha = boost::lexical_cast<float>(argv[1]);
    int num_topic = boost::lexical_cast<int>(argv[2]);
    string model_file = argv[3];
    string query = argv[4];

    int max_step = 10;

    LdaModel* p_lda_model = LdaModel::get_instance(model_file, num_topic, alpha);

    RtLdaPredictor predictor(p_lda_model);

    vector<int> input;
    istringstream iss(query);
    string buf;
    while (iss>>buf)
    {
        input.push_back(boost::lexical_cast<int>(buf));
    }
    
    vector<int> output;
    predictor.predict(input, max_step, output);
    return 0;
}
