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

double mhz = 2128.054 * 1000;  // 核的频率，在/proc/cpuinfo中找



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

    LDAQueryExtend lda_query_extender(model_file, alpha, 0.0, 10, 50); 
    unordered_map<string, double> extended_query;

    long long start, end;
    start = get_cycles();
    lda_query_extender.ExtendQuery(tokens, &extended_query); 
    end = get_cycles();
    double millisecond = (end - start) / mhz;

    cout<<"cost "<<millisecond<<" ms"<<endl;
    cout<<"------ extended query --------"<<endl;    
    vector<pair<string, double> > rst(extended_query.begin(), extended_query.end());
    sort(rst.begin(), rst.end(), cmper()); 
    cout<<rst<<endl;
    
    return 0;
}
