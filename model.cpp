#include "model.h"


int main(int argc, char *argv[])
{
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

    Model model(model_file);
    Sampler sampler(&model, alpha, 0.0);
    LdaInfer infer(&sampler);

    Document doc;
    infer.Infer(tokens, 20, &doc);
    
    for (size_t i = 0; i < doc._document.size(); ++i)
        cout<<doc._document[i]<<":"<<doc._topic[i]<<endl;

    return 0;
}
