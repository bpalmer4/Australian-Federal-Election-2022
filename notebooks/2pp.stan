// STAN: Two-Party Preferred (TPP) Vote Intention Model 

data {
    // data size
    int<lower=1> n_polls;
    int<lower=1> n_days;
    int<lower=1> n_houses;
    
    // assumed standard deviation for all polls
    real<lower=0> pseudoSampleSigma;
    
    // poll data
    vector<lower=0,upper=100>[n_polls] y; // TPP vote share
    int<lower=1> house[n_polls];
    int<lower=1> day[n_polls];
}

transformed data {
    real<lower=0,upper=100> centre_offset = mean(y);
    vector<lower=-100,upper=100> [n_polls] yy = y - centre_offset;
    real sigma = 0.15; // day-to-day std deviation in percentage points 
}

parameters {
    vector[n_days] hidden_vote_share; 
    vector[n_houses] pHouseEffects;
}

transformed parameters {
    // -- sum to zero constraint on house effects
    vector[n_houses] houseEffect;
    houseEffect[1:n_houses] = pHouseEffects[1:n_houses] - mean(pHouseEffects[1:n_houses]);
}

model {
    // -- temporal model [this is the hidden state space model]
    hidden_vote_share[1] ~ cauchy(centre_offset, 10); // PRIOR (reasonably uninformative)
    hidden_vote_share[2:n_days] ~ normal(hidden_vote_share[1:(n_days-1)], sigma);
    
    // -- house effects model
    pHouseEffects ~ cauchy(0, 10); // PRIOR (reasonably uninformative)

    // -- observed data / measurement model
    yy ~ normal(houseEffect[house] + hidden_vote_share[day] - centre_offset, pseudoSampleSigma);
}
