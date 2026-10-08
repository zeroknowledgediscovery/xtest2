"""Patch upstream native LSmash to expose stochastic projector likelihoods."""
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
assert 'from_sequences_random_projectors' not in s
s='#include <random>\n'+s
code=r'''
    m.def("from_sequences_random_projectors", [](const std::vector<std::vector<unsigned int>>& seqs,
            int n_probes, int n_states, unsigned int seed, bool crosscheck) {
        if (n_probes < 1 || n_states < 2) throw std::invalid_argument("invalid probe count/state count");
        std::vector<symbol_list_> paths=to_symbol_lists(seqs);
        std::mt19937 mt(seed);
        std::uniform_int_distribution<int> next_state(0,n_states-1);
        std::uniform_real_distribution<double> logits(-1.5,1.5);
        std::vector<PFSA> machines;
        machines.reserve(n_probes);
        for(int k=0;k<n_probes;++k) {
            pitilde emission;
            connx transitions;
            for(int q=0;q<n_states;++q) {
                double t=logits(mt);
                double p1=1./(1.+std::exp(-t));
                emission[q]={1.-p1,p1};
                transitions[q][symbol(0)]=(q+1)%n_states;
                transitions[q][symbol(1)]=next_state(mt);
            }
            machines.emplace_back(emission,transitions);
        }
        py::array_t<double> feats({static_cast<py::ssize_t>(seqs.size()),static_cast<py::ssize_t>(n_probes)});
        auto out=feats.mutable_unchecked<2>();
        for(int j=0;j<n_probes;j++) {
            std::vector<double> f=machines[j].log_likelihood(paths);
            for(size_t i=0;i<f.size();++i)out(i,j)=f[i];
        }
        if(crosscheck && seqs.size()<=20) {
            matrix_dbl dist=llk_distance(paths,machines,2);
            for(size_t i=0;i<paths.size();i++)for(size_t j=0;j<paths.size();j++) {
                double predicted=0.;
                for(int k=0;k<n_probes;k++)predicted+=std::abs(out(i,k)-out(j,k));
                predicted/=n_probes;
                if(std::abs(predicted-dist[i][j])>1.e-8) throw std::runtime_error("LSmash formula mismatch");
            }
        }
        return feats;
    },py::arg("seqs"),py::arg("n_probes"),py::arg("n_states"),py::arg("seed"),py::arg("crosscheck")=false,
    "Native PFSA log-likelihood projections; pairwise mean absolute differences equal LSmash distances.");
'''
i=s.rfind('\n}')
assert i>s.index('PYBIND11_MODULE(_lsmash, m) {')
s=s[:i]+code+s[i:];p.write_text(s)
print("Patched native LSmash C++ extension",p)
