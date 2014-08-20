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
#include <glog/logging.h>
#include <stdio.h>
#include <map>
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
    string document_output_string()
    {
        string output;
        char buf[100];
        for (size_t i = 0; i < _document.size(); ++i)
        {
            snprintf(buf, sizeof(buf), "%d:%d ", _document[i], _topic[i]);
            output += buf;
        }
        return output;
    }
};

void print_topic_count_dist(const TopicCountDist& topic_count_dist)
{
    for (TopicCountDist::const_iterator iter = topic_count_dist.begin();
         iter != topic_count_dist.end();
         ++iter)
        cout<<"topic_id:"<<iter->first<<" : "<<iter->second<<endl;
}

class Model {
public:
    typedef unordered_map<int, TopicCountDist* >  WordTopicCountDist;
    typedef unordered_map<string, int> Word2IDDict;
    typedef unordered_map<int, string> ID2WordDict;
    
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

                if (_word2id_dict.find(word) == _word2id_dict.end())
                {
                    int size = _word2id_dict.size();
                    _word2id_dict[word] = size;
                    _id2word_dict[size] = word;
                } 
                int word_id = _word2id_dict[word];
                if (_wor2top.find(word_id) == _wor2top.end())
                {
                    _wor2top[word_id] = new TopicCountDist();
                }
                (*_wor2top[word_id])[topic_id] = count;

                topic_total_count += count; 
            }

            _top_total[topic_id] = topic_total_count;
        }
        LOG(INFO)<<"Load Model over: num_topic="<<_top_total.size()
                 <<" num_vocal="<<_word2id_dict.size()<<endl;
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
        return _word2id_dict.size();
    }

    inline Word2IDDict* GetWord2IDDict()
    {
        return &_word2id_dict;
    }

    inline ID2WordDict* GetID2WordDict()
    {
        return &_id2word_dict;
    }

    ~Model()
    {    
        for (WordTopicCountDist::iterator iter = _wor2top.begin();
             iter != _wor2top.end();
             ++iter)
        {
            delete iter->second;
            iter->second = NULL;
        }
    }

private:
    WordTopicCountDist _wor2top;
    TopicCountDist _top_total;
    Word2IDDict _word2id_dict;
    ID2WordDict _id2word_dict;
};




class LdaInfer {
public:
    LdaInfer(string model_file, double alpha, double beta) 
    : _model(model_file), _alpha(alpha), _beta(beta)
    {    _num_topic = _model.GetTopicNum(); }

    void Infer(const vector<string>& string_doc, int max_iter, vector<pair<string, int> >* word_topic)
    {
        Document doc;
        InitTopicAssignment(string_doc, &doc);
        for (int iter = 0; iter < max_iter; ++iter)
        {
            UpdateTopicForDocument(&doc);
        }

        Model::ID2WordDict* id2word_dict = _model.GetID2WordDict();
        for (size_t i=0; i<doc._document.size(); ++i)
        {
            int word_id = doc._document[i];
            int topic_id = doc._topic[i];
            string word = (*id2word_dict)[word_id]; 
            word_topic->push_back(pair<string, int>(word, topic_id));
        }
    }

private:
    void UpdateTopicForDocument(Document* doc)
    {
        //cout<<"-------------------------------------------------------"<<endl;
        //LOG(INFO)<<"Before Update Topic: "<<doc->document_output_string()<<endl;
        //print_topic_count_dist(doc->_topic_dist);
        //cout<<"-------------------------------------------------------"<<endl;

        int doc_size = doc->_document.size();
        for (size_t i = 0; i < doc_size; ++i)
        {
            int old_topic_id = doc->_topic[i];
            TopicCountDist topic_count_dist;
            CalcTopicPosterior(i, doc, &topic_count_dist);
            int sampled_topic = SampleTopic(&topic_count_dist);
            //LOG(INFO)<<"word "<<doc->_document[i]<<" sampled topic is "<<sampled_topic<<endl;
            // update topic assignment
            doc->_topic[i] = sampled_topic;
            if (doc->_topic_dist.find(sampled_topic) == doc->_topic_dist.end())
            {
                doc->_topic_dist[sampled_topic] = 1;
            }
            else {
                doc->_topic_dist[sampled_topic] += 1;
            }
            doc->_topic_dist[old_topic_id] -= 1;
        }
    }

    // p(z|w, \theta, \phi, alpha), we omit beta here cause it is not important
    void CalcTopicPosterior(int word_id_index, Document* doc, TopicCountDist* topic_count_dist)
    {
        int word_id = doc->_document[word_id_index];
        int old_topic_id = doc->_topic[word_id_index];
        const TopicCountDist* word_topic_count_dist = _model.GetWordTopicCountDist(word_id);
        for (TopicCountDist::const_iterator iter = word_topic_count_dist->begin();
             iter != word_topic_count_dist->end();
             ++iter)
        {
           int topic_id = iter->first;
           double topic_count = iter->second;
           double topic_total_count = _model.GetTopicTotalCount(topic_id);
           double p_w_z = topic_count / topic_total_count;

           double adjust = topic_id == old_topic_id ? 1 : 0;
           double doc_topic_count = doc->_topic_dist.find(topic_id) != doc->_topic_dist.end()\
                                    ? doc->_topic_dist[topic_id] : 0.0;  
           doc_topic_count -= adjust;
           
           (*topic_count_dist)[topic_id] = p_w_z * (doc_topic_count + _alpha);
           //cout<<"+++posterior topic:"<<topic_id
           //    <<" prob:"<<(*topic_count_dist)[topic_id]
           //    <<" p_w_z:"<<p_w_z<<"  p_z:"<<(doc_topic_count + _alpha)<<endl;
        }
    }

    int SampleTopic(const TopicCountDist* topic_count_dist)
    {
        vector<TopicCountPair > topic_dist(topic_count_dist->begin(), topic_count_dist->end());
        size_t size = topic_dist.size();
        for (size_t i = 1; i < size; ++i)
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

            Model::Word2IDDict* p_word2id_dict = _model.GetWord2IDDict();
            if (p_word2id_dict->find(string_doc[i]) == p_word2id_dict->end()) continue;
            int word_id = (*p_word2id_dict)[string_doc[i]];
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
    //Sampler _sampler; 
    Model _model;
    int _num_topic;
    double _alpha;
    double _beta;
};

class LDAQueryExtend {
public:
    typedef pair<string, double> WordProb;
    LDAQueryExtend(const string& model_file, double alpha, double beta)
    : _infer(model_file, alpha, beta), _topic2word(6000)
    {
        ifstream ifs(model_file.c_str());
        string buf;
        int topic2word_size = _topic2word.size();
        while (getline(ifs, buf))
        {
            if (buf.size() == 0 && buf[0] == '\n') continue;
            istringstream ss(buf);
            int topic_id;
            ss >> topic_id;
            if (topic2word_size <= topic_id) {
                _topic2word.resize(topic2word_size);
                topic2word_size = _topic2word.size();
            }
            
            string item;
            double total_count = 0.0;
            while (ss >> item)
            {
                vector<string> tokens;
                boost::split(tokens, item, boost::is_any_of(":"));
                string word = tokens[0];
                double count = boost::lexical_cast<double>(tokens[1]);
                total_count += count;
                _topic2word[topic_id].push_back(WordProb(word, count));
            }
            
            for (size_t i = 0; i<_topic2word[topic_id].size(); ++i)
                _topic2word[topic_id][i].second /= total_count;
        }

    }

    void ExtendQuery(const vector<string>& tokens, unordered_map<string, double>* extended_query)
    {
        vector<pair<string, int> > word_topic;
        _infer.Infer(tokens, 20, &word_topic);
        int sz = word_topic.size();
        TopicCountDist topic_dist;
        for (size_t i=0; i<sz; ++i)
        {
            int topic_id = word_topic[i].second;
            if (topic_dist.find(topic_id) == topic_dist.end())
                topic_dist[topic_id] = 1.0/sz;
            else
                topic_dist[topic_id] += 1.0/sz;
            cout<<word_topic[i].first<<":"<<word_topic[i].second<<endl;
        }

        for (TopicCountDist::iterator iter = topic_dist.begin();
             iter != topic_dist.end();
             ++iter)
        {
            int topic_id = iter->first;
            double prob_topic = iter->second;
            
            vector<WordProb>& one_topic = _topic2word[topic_id];
            int topic_word_size = one_topic.size();
            for (size_t i=0; i<topic_word_size; ++i)
            {
                string word = one_topic[i].first;
                double prob_topic2word = one_topic[i].second;
                if (extended_query->find(word) == extended_query->end())
                    (*extended_query)[word] = prob_topic * prob_topic2word;
                else
                    (*extended_query)[word] += prob_topic * prob_topic2word;
            }
        }
    }

private:
    vector<vector<WordProb > > _topic2word; 
    LdaInfer _infer;
};

#endif
