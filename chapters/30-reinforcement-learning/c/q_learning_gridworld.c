/*
 * Chapter 30 - tabular Q-learning from scratch in pure C.
 *
 * The complete reinforcement-learning algorithm, no framework, no library:
 * an agent learns to cross a gridworld to a goal (avoiding a pit) knowing
 * only the rewards it receives. The Q-table is a plain array; the update is
 * one line; the result is a learned policy that you can print as a map of
 * arrows. This is the same algorithm as the Python version's part 1, and the
 * conceptual core of the DQN that follows it there.
 *
 * Build and run from the repository root:
 *     make -C chapters/30-reinforcement-learning/c
 *     ./chapters/30-reinforcement-learning/c/build/q_learning_gridworld
 */

#include <stdint.h>
#include <stdio.h>

#define GRID_ROWS 3
#define GRID_COLUMNS 4
#define ACTION_COUNT 4    /* up, down, left, right */

static const int action_deltas[ACTION_COUNT][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
static const char action_symbols[ACTION_COUNT] = {'^', 'v', '<', '>'};

/* Q[row][column][action] = learned value of taking that action there. */
static double q_table[GRID_ROWS][GRID_COLUMNS][ACTION_COUNT];

/* Chapter 9's deterministic generator, so training is reproducible. */
static uint64_t random_state = 42;
static double next_uniform(void) {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(random_state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Take one step in the gridworld.
 *
 * row, column:      current position (updated in place to the next position)
 * action:           0-3
 * reward_out:       receives the reward for this step
 *
 * Returns 1 if the episode ended (goal or pit), else 0. The goal is (0,3)
 * with +1, the pit (1,3) with -1; every other step costs a little (-0.02) so
 * the agent learns to hurry.
 */
static int grid_step(int *row, int *column, int action, double *reward_out) {
    int next_row = *row + action_deltas[action][0];
    int next_column = *column + action_deltas[action][1];
    if (next_row < 0) next_row = 0;
    if (next_row >= GRID_ROWS) next_row = GRID_ROWS - 1;
    if (next_column < 0) next_column = 0;
    if (next_column >= GRID_COLUMNS) next_column = GRID_COLUMNS - 1;
    *row = next_row;
    *column = next_column;

    if (next_row == 0 && next_column == 3) { *reward_out = 1.0; return 1; }   /* goal */
    if (next_row == 1 && next_column == 3) { *reward_out = -1.0; return 1; }  /* pit */
    *reward_out = -0.02;
    return 0;
}

static int best_action(int row, int column) {
    int best = 0;
    for (int a = 1; a < ACTION_COUNT; a++) {
        if (q_table[row][column][a] > q_table[row][column][best]) best = a;
    }
    return best;
}

int main(void) {
    const double discount = 0.95;
    const double learning_rate = 0.1;
    double exploration_rate = 1.0;

    for (int episode = 0; episode < 3000; episode++) {
        int row = 2, column = 0;   /* start bottom-left */
        for (int step = 0; step < 50; step++) {
            /* epsilon-greedy: explore randomly sometimes, else act greedily. */
            int action;
            if (next_uniform() < exploration_rate) {
                action = (int)(next_uniform() * ACTION_COUNT);
                if (action >= ACTION_COUNT) action = ACTION_COUNT - 1;
            } else {
                action = best_action(row, column);
            }

            int previous_row = row, previous_column = column;
            double reward;
            int done = grid_step(&row, &column, action, &reward);

            /* The Q-learning update - the whole algorithm in one line:
             * move the estimate toward (reward now) + (best achievable next). */
            double best_next = 0.0;
            if (!done) {
                best_next = q_table[row][column][best_action(row, column)];
            }
            double temporal_difference = reward + discount * best_next
                                       - q_table[previous_row][previous_column][action];
            q_table[previous_row][previous_column][action] += learning_rate * temporal_difference;

            if (done) break;
        }
        if (exploration_rate > 0.05) exploration_rate *= 0.995;
    }

    printf("Tabular Q-learning solved the gridworld from rewards alone.\n\n");
    printf("Learned policy (best action per cell):\n");
    for (int row = 0; row < GRID_ROWS; row++) {
        printf("   ");
        for (int column = 0; column < GRID_COLUMNS; column++) {
            if (row == 0 && column == 3) printf(" GOAL ");
            else if (row == 1 && column == 3) printf(" PIT  ");
            else printf("  %c   ", action_symbols[best_action(row, column)]);
        }
        printf("\n");
    }

    printf("\nEvery arrow points along the shortest safe route to the goal - and the agent\n");
    printf("was never shown that route. It discovered it by trying actions and remembering\n");
    printf("which led to reward. That is reinforcement learning, in one file, no framework.\n");
    return 0;
}
