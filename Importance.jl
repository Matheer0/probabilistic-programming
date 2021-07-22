using Gen
using PyPlot

@gen function flip_biased_coin(N)
    θ ~ beta(1,1)
    
    [{:flip => i} ~ bernoulli(θ)  for i in 1:N]
end;

observations = Gen.choicemap()
(t, w) = Gen.generate(flip_biased_coin,(3,),observations)
Gen.get_choices(t)

function iter_deep(c::Gen.ChoiceMap)
  Iterators.flatten([
      Gen.get_values_shallow(c),
      (Dict(Pair{Any, Any}(k => kk, vv) for (kk, vv) in iter_deep(v))
       for (k, v) in Gen.get_submaps_shallow(c))...,
  ])
end

function check_conditions(trace, constraints)
    for (name, value) in iter_deep(constraints)
        if trace[name] != value return false end
    end
    return true
end

function rejection(generative_model, arguments, constraints) 
    (t,w)=Gen.generate(generative_model,arguments,constraints)
    if check_conditions(t,constraints)
        return(t)
    else
        rejection(generative_model, arguments, constraints)
    end
end

observations = Gen.choicemap()

for i in 1:10000
    observations[:flip => i] = true
end

valid_trace=rejection(flip_biased_coin, (10010,), observations);
Gen.get_choices(valid_trace)

observations = Gen.choicemap()

for i in 1:10000
    observations[:flip => i] = true
end

(valid_trace, weight)=Gen.generate(flip_biased_coin, (20000,), observations);
Gen.get_choices(valid_trace)

observations = Gen.choicemap()

for i in 1:1000
    observations[:flip => i] = true
end

hist([Gen.generate(flip_biased_coin, (2000,), observations)[1][:θ] for _ in 1:10000], bins=100);

observations = Gen.choicemap()

for i in 1:1000
    observations[:flip => i] = true
end

function weighted_sample()
    (t,w) = Gen.generate(flip_biased_coin, (2000,), observations)
    (t[:θ],exp(w))
end
thetas, weights = zip([weighted_sample() for _ in 1:100]...)
scatter(thetas, weights);

function my_importance_resampling(model,arguments,constraints,computation)
    samples=[Gen.generate(model,arguments,constraints) for i=1:computation]
    ps=map(x -> exp(x[2]), samples)
    ps_norm=ps/sum(ps)
    log_ps_average=log(sum(ps)/length(ps))
    result=samples[categorical(ps_norm)]
    (result[1], log_ps_average)
end

observations = Gen.choicemap()

num=100
for i in 1:num
    observations[:flip => i] = true
end

ss=[my_importance_resampling(flip_biased_coin,(num+ceil(.5*num),),observations,50) for _ in 1:100]
θs = ps=map(x -> x[1][:θ], ss)
hist(θs);
