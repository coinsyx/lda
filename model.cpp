#include "model.h"


struct cmper {
    bool operator()(const pair<string, double>& lhs, const pair<string, double>& rhs)
    {
        return lhs.second > rhs.second;
    }
};

ostream& operator<<(ostream& out, vector<pair<string, double> >& v)
{
    for (size_t i=0; i<v.size(); i++)
        out<<v[i].first<<":"<<v[i].second<<endl;
    return out;
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    if (argc < 2)
    {
        cout<<"Usage: "<<argv[0]<<"model_file alpha input_query"<<endl;
        return 0;
    }

    string model_file = argv[1];
    double alpha = boost::lexical_cast<double>(argv[2]);
    string query = argv[3];

    vector<string> tokens;
    boost::split(tokens, query, boost::is_any_of(" "));

    //Sampler sampler(&model, alpha, 0.0);
    //LdaInfer infer(model_file, alpha, 0.0);
    LDAQueryExtend lda_query_extender(model_file, alpha, 0.0); 
    unordered_map<string, double> extended_query;
    lda_query_extender.ExtendQuery(tokens, &extended_query); 
    cout<<"------ extended query --------"<<endl;    
    vector<pair<string, double> > rst(extended_query.begin(), extended_query.end());
    sort(rst.begin(), rst.end(), cmper()); 
    cout<<rst<<endl;
    return 0;
}
