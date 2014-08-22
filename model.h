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


template<typename KeyType>
inline void IncreaseKeyCount(unordered_map<KeyType, double>* hashmap, KeyType key, double value)
{
    if (hashmap->find(key) != hashmap->end())
        (*hashmap)[key] += value;
    else
        (*hashmap)[key] = value;
}

struct Document
{
    vector<string> _string_document;  // word_string vector
    vector<int> _document;            // word_id vector
    vector<int> _topic;               // corresponding topic id
    TopicCountDist _topic_dist;       // topic count in document
    TopicCountDist _accumulate_topic_dist;  // accumulated topic count since after burn-in
    vector<string> _unknown_word;     // words unseen by model
};

class Model {
public:
    typedef unordered_map<int, TopicCountDist* >  WordTopicCountDist;
    typedef unordered_map<string, int> Word2IDDict;
    
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
};




class LdaInfer {
public:
    LdaInfer(string model_file, double alpha, double beta, int burnin_iter, int max_iter) 
    : _model(model_file), _alpha(alpha), _beta(beta), _burnin_iter(burnin_iter), _max_iter(max_iter)
    {    _num_topic = _model.GetTopicNum(); }

    void Infer(const vector<string>& string_doc, Document* doc)
    {
        //Document doc;
        InitTopicAssignment(string_doc, doc);
        if (doc->_document.size() == 0)  return;

        int accumulate_count = _max_iter - _burnin_iter;
        int doc_len = doc->_document.size();
        TopicCountDist& accumulate_topic_dist = doc->_accumulate_topic_dist;
        for (int n = 0; n < _max_iter; ++n)
        {
            UpdateTopicForDocument(doc);
            //accumulate topic count
            if ( n >= _burnin_iter)
            {
                for (TopicCountDist::iterator iter = doc->_topic_dist.begin();
                     iter != doc->_topic_dist.end();
                     ++iter)
                {
                    IncreaseKeyCount(&(doc->_accumulate_topic_dist), 
                                     iter->first, 
                                     iter->second/ (accumulate_count*doc_len));
                }
            }
        }
    }

private:
    void UpdateTopicForDocument(Document* doc)
    {
        int doc_size = doc->_document.size();
        for (size_t i = 0; i < doc_size; ++i)
        {
            // calculate topic posterior
            TopicCountDist topic_count_dist;
            CalcTopicPosterior(i, doc, &topic_count_dist);
            // sample from topic distribution
            int sampled_topic = SampleTopic(&topic_count_dist);
            // update topic assignment
            IncreaseKeyCount(&(doc->_topic_dist), doc->_topic[i], -1);
            doc->_topic[i] = sampled_topic;
            IncreaseKeyCount(&(doc->_topic_dist), sampled_topic, 1);
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
           TopicCountDist& topic_dist = doc->_topic_dist;
           double doc_topic_count = topic_dist.find(topic_id) != topic_dist.end()\
                                    ? topic_dist[topic_id] : 0.0;  
           doc_topic_count -= adjust;
           
           (*topic_count_dist)[topic_id] = p_w_z * (doc_topic_count + _alpha);
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
            if (p_word2id_dict->find(string_doc[i]) == p_word2id_dict->end())
            {
                doc->_unknown_word.push_back(string_doc[i]);
                continue;
            }
            doc->_string_document.push_back(string_doc[i]);
            int word_id = (*p_word2id_dict)[string_doc[i]];
            doc->_document.push_back(word_id);
            int random_topic = static_cast<int>( rand() / static_cast<double>(RAND_MAX) * _num_topic);
            doc->_topic.push_back(random_topic);

            IncreaseKeyCount(&(doc->_topic_dist), random_topic, 1);
        }
    }

private:
    Model _model;
    int _num_topic;
    double _alpha;
    double _beta;
    int _max_iter;
    int _burnin_iter;
};

class LDAQueryExtend {
public:
    typedef pair<string, double> WordProb;
    LDAQueryExtend(const string& model_file, double alpha, double beta, int burnin_iter, int max_iter)
    : _infer(model_file, alpha, beta, burnin_iter, max_iter), _topic2word(6000)
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
                _topic2word.resize(topic2word_size * 2);
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
        Document doc;
        _infer.Infer(tokens, &doc);
        build_extended_query(doc, extended_query);

        cout<<"------ topic distribution -------"<<endl;
        for (TopicCountDist::iterator iter = doc._accumulate_topic_dist.begin();
             iter != doc._accumulate_topic_dist.end();
             ++iter)
            if (iter->second > 1e-4) cout<<iter->first<<":"<<iter->second<<endl;
        cout<<"---------------------------------"<<endl;
    }

    void build_extended_query(Document& doc, unordered_map<string, double>* extended_query)
    {
        double topic_dist_weight = 0.5;
        TopicCountDist& topic_dist = doc._accumulate_topic_dist;
        // extend query
        for (TopicCountDist::iterator iter = topic_dist.begin();
             iter != topic_dist.end();
             ++iter)
        {
            int topic_id = iter->first;
            double prob_topic = iter->second;
            if (prob_topic < 1e-4)  continue; // skip unlikely topic 
            vector<WordProb>& one_topic = _topic2word[topic_id];
            int topic_word_size = one_topic.size();
            for (size_t i=0; i<topic_word_size; ++i)
            {
                const string& word = one_topic[i].first;
                double prob_topic2word = one_topic[i].second;
                IncreaseKeyCount(extended_query, word, prob_topic * prob_topic2word * topic_dist_weight);
            }
        }

        // modify orignal query weight, known words
        int sz = doc._string_document.size();
        int doc_len = doc._string_document.size() + doc._unknown_word.size();
        for (size_t i=0; i<sz; ++i)
        {
            const string& word = doc._string_document[i];
            IncreaseKeyCount(extended_query, word, 1.0 / doc_len * (1-topic_dist_weight));
        }
        // modify orignal query weight, known words
        sz = doc._unknown_word.size();
        for (size_t i=0; i<sz; ++i)
        {
            const string& word = doc._unknown_word[i];
            IncreaseKeyCount(extended_query, word, 1.0 / doc_len * (1-topic_dist_weight));
        }
    }

private:
    vector<vector<WordProb > > _topic2word; 
    LdaInfer _infer;
};

#endif
