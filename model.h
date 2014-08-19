#ifndef MODEL_H_
#define MODEL_H_

#include <tr1/unordered_map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <functional>
#include <ext/functional>
using namespace std;
using namespace __gnu_cxx;
using tr1::unordered_map;

typedef unordered_map<int, double>  TopicCountDist;
typedef pair<int, double> TopicCountPair;

struct Document
{
    vector<int> _document;
    vector<int> _topic;
    TopicCountDist _topic_dist;    
};


class Model {
public:
    typedef unordered_map<int, TopicCountDist* >  WordTopicCountDist;
    // word -> wordid
    typedef unordered_map<string, int> WordDict;
    
    Model (const string& model_file)
    {
        // topic_id \t word:count \t word:count ...
        ifstream ifs(model_file.c_str());
        string buf;
        int n = 0;
        while (getline(ifs, buf))
        {
            if (buf.size() == 0 && buf[0] == '\n') continue;

            istringstream ss(buf);
            int topic_id;
            ss >> topic_id;
            string item;
            double topic_total_count = 0;
            while (ss>>item)
            {
                vector<string> tokens;
                boost::split(tokens, item, boost::is_any_of(":"));
                string word = tokens[0];
                double count = boost::lexical_cast<double>(tokens[1]);

                if (_word_dict.find(word) == _word_dict.end())
                {
                    _word_dict[word] = _word_dict.size();
                } 
                int word_id = _word_dict[word];
                if (_wor2top.find(word_id) == _wor2top.end())
                {
                    _wor2top[word_id] = new TopicCountDist();
                }
                (*_wor2top[word_id])[topic_id] = count;

                topic_total_count += count; 
            }

            _top_total[topic_id] = topic_total_count;
        }
    }

    inline const TopicCountDist* GetWordTopicCountDist (int word_id)
    {
        return _wor2top[word_id];
    }

    inline double GetTopicTotalCount(int topic_id)
    {
        return _top_total[topic_id];
    }

    inline int GetTopicNum()
    {
        return _top_total.size();
    }
    inline int GetVocalNum()
    {
        return _word_dict.size();
    }

    inline WordDict* GetWordDict()
    {
        return &_word_dict;
    }

    ~Model()
    {    }

private:
    WordTopicCountDist _wor2top;
    TopicCountDist _top_total;
    WordDict _word_dict;
};


class Sampler {
public:
    Sampler(Model* model, double alpha, double beta) 
    : _model(model), _alpha(alpha), _beta(beta) 
    {
        _num_topic = _model->GetTopicNum();
        _num_vocal = _model->GetVocalNum();
    }

    void UpdateTopicForDocument(Document* doc)
    {
        int doc_size = doc->_document.size();
        for (size_t i = 0; i < doc_size; ++i)
        {
            TopicCountDist topic_count_dist;
            CalcTopicPosterior(i, doc, &topic_count_dist);
            int sampled_topic = SampleTopic(&topic_count_dist);
            // update topic assignment
            doc->_topic[i] = sampled_topic;
            if (doc->_topic_dist.find(sampled_topic) == doc->_topic_dist.end()
                || doc->_topic_dist[sampled_topic] == 0)
            {
                doc->_topic_dist[sampled_topic] = 1;
            }
            else {
                doc->_topic_dist[sampled_topic] -= 1;
            }
        }
    }

    // p(z|w, \theta, \phi, alpha), we omit beta here cause it is not important
    void CalcTopicPosterior(int word_id_index, Document* doc, TopicCountDist* topic_count_dist)
    {
        int word_id = doc->_document[word_id_index];
        int old_topic_id = doc->_topic[word_id_index];
        const TopicCountDist* word_topic_count_dist = _model->GetWordTopicCountDist(word_id);
        for (TopicCountDist::const_iterator iter = word_topic_count_dist->begin();
             iter != word_topic_count_dist->end();
             ++iter)
        {
           int topic_id = iter->first;
           double topic_count = iter->second;
           double topic_total_count = _model->GetTopicTotalCount(topic_id);
           double p_w_z = topic_count / topic_total_count;

           double adjust = topic_id == old_topic_id ? 1 : 0;
           double doc_topic_count = doc->_topic_dist.find(topic_id) != doc->_topic_dist.end()\
                                    ? doc->_topic_dist[topic_id] : 0.0;  
           doc_topic_count -= adjust;
           
           (*topic_count_dist)[topic_id] = p_w_z * (doc_topic_count + _alpha);
        }
    }

    int SampleTopic(const TopicCountDist* topic_count_dist)
    {
        vector<TopicCountPair > topic_dist(topic_count_dist->begin(), topic_count_dist->end());
        size_t size = topic_dist.size();
        for (int i = 1; i < size; ++i)
            topic_dist[i].second += topic_dist[i-1].second;
        double rdm = rand() / static_cast<double>(RAND_MAX) * topic_dist[size-1].second;
        vector<TopicCountPair >::iterator iter = find_if(topic_dist.begin(), topic_dist.end(),
            compose1(bind1st(less_equal<double>(), rdm), _Select2nd<TopicCountPair >()));
        return iter->first;
    }

    int InitTopicAssignment(const vector<string>& string_doc, Document* doc)
    {
        for (size_t i = 0; i < string_doc.size(); ++i)
        {

            Model::WordDict* p_word_dict = _model->GetWordDict();
            if (p_word_dict->find(string_doc[i]) == p_word_dict->end()) continue;
            int word_id = (*p_word_dict)[string_doc[i]];
            doc->_document.push_back(word_id);
            int random_topic = static_cast<int>( rand() / static_cast<double>(RAND_MAX) * _num_topic);
            doc->_topic.push_back(random_topic);

            if (doc->_topic_dist.find(random_topic) == doc->_topic_dist.end())
                doc->_topic_dist[random_topic] = 1.0;
            else
                doc->_topic_dist[random_topic] ++;
        }
    }
private:
    Model* _model;
    int _num_topic;
    int _num_vocal;
    double _alpha;
    double _beta;
};


class LdaInfer {
public:
    LdaInfer(Sampler* sampler) : _sampler(sampler) { }

    void Infer(const vector<string>& string_doc, int max_iter, Document* doc)
    {
        _sampler->InitTopicAssignment(string_doc, doc);
        for (int iter = 0; iter < max_iter; ++iter)
        {
            _sampler->UpdateTopicForDocument(doc);
        }
    }

private:
    Sampler* _sampler; 
};





#endif
