%%
transitionMatrix = [0.9 0.1; 0.1 0.9];
initialProbabilityState = [1; 0]; % Made it a column vector, rather than a row.
nStates = 100;
states = zeros(2,nStates);
states(:,1) = initialProbabilityState;
for ns = 2:nStates
    states(:,ns) = transitionMatrix*states(:,ns-1);
end
states

%%
transition_probabilities = [0.1 0.9;0.8 0.2]; starting_value = 1; chain_length = 100;
    chain = zeros(1,chain_length);
    chain(1)=starting_value;
    for i=2:chain_length
        this_step_distribution = transition_probabilities(chain(i-1),:);
        cumulative_distribution = cumsum(this_step_distribution);
        r = rand();
        chain(i) = find(cumulative_distribution>r,1);
    end