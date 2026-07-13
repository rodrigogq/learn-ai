/*
 * Chapter 8 - a tiny automatic differentiation engine in pure C, then XOR
 * learned with it. Full port of the Python example, same numbers.
 *
 * The structural difference from the Python version (explained in the
 * chapter): C has no operator overloading and no closures, so the graph is an
 * ARENA - one flat array of node structs. Each node stores its value, its
 * gradient, which operation created it, and the array indices of its parents.
 * Creation order is already a valid topological order (parents always exist
 * before children), so the backward pass is a plain reverse loop with a
 * switch on the operation type.
 *
 * Build and run from the repository root:
 *     make -C chapters/08-backpropagation/c
 *     ./chapters/08-backpropagation/c/build/tiny_autograd
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MAXIMUM_GRAPH_NODES 1024

typedef enum {
    OPERATION_LEAF,      /* an input or parameter: nothing to propagate */
    OPERATION_ADD,
    OPERATION_MULTIPLY,
    OPERATION_TANH,
} OperationType;

typedef struct {
    double data;          /* forward-pass value */
    double gradient;      /* dLoss/d(this node), filled by the backward pass */
    OperationType operation;
    int first_parent;     /* index into the arena, -1 if unused */
    int second_parent;
} GraphNode;

static GraphNode graph_nodes[MAXIMUM_GRAPH_NODES];
static int graph_node_count = 0;

/*
 * Create a leaf node (an input or a trainable parameter).
 *
 * data: the number the leaf holds.
 *
 * Returns the new node's arena index. All node-creating functions exit the
 * program if the arena is full - simpler than error codes for a teaching tool.
 */
static int create_leaf(double data) {
    if (graph_node_count >= MAXIMUM_GRAPH_NODES) {
        fprintf(stderr, "graph arena full\n");
        exit(1);
    }
    int node_index = graph_node_count++;
    graph_nodes[node_index] = (GraphNode){data, 0.0, OPERATION_LEAF, -1, -1};
    return node_index;
}

/*
 * Record an addition node: result = nodes[first] + nodes[second].
 *
 * first_parent, second_parent: arena indices of the two values being added.
 */
static int add_values(int first_parent, int second_parent) {
    int node_index = create_leaf(graph_nodes[first_parent].data + graph_nodes[second_parent].data);
    graph_nodes[node_index].operation = OPERATION_ADD;
    graph_nodes[node_index].first_parent = first_parent;
    graph_nodes[node_index].second_parent = second_parent;
    return node_index;
}

/*
 * Record a multiplication node: result = nodes[first] * nodes[second].
 *
 * first_parent, second_parent: arena indices of the two values being multiplied.
 */
static int multiply_values(int first_parent, int second_parent) {
    int node_index = create_leaf(graph_nodes[first_parent].data * graph_nodes[second_parent].data);
    graph_nodes[node_index].operation = OPERATION_MULTIPLY;
    graph_nodes[node_index].first_parent = first_parent;
    graph_nodes[node_index].second_parent = second_parent;
    return node_index;
}

/*
 * Record a tanh node: result = tanh(nodes[parent]).
 *
 * parent: arena index of the value being squashed.
 */
static int tanh_value(int parent) {
    int node_index = create_leaf(tanh(graph_nodes[parent].data));
    graph_nodes[node_index].operation = OPERATION_TANH;
    graph_nodes[node_index].first_parent = parent;
    return node_index;
}

/*
 * Backpropagation: seed dL/dL = 1 at the loss node, then walk the arena in
 * reverse creation order applying each node's local derivative rule.
 *
 * loss_node: arena index of the final value (the loss).
 *
 * Gradients ACCUMULATE with += because a value used in several places
 * collects gradient from every path (the chapter's "one extra rule").
 */
static void run_backward_pass(int loss_node) {
    graph_nodes[loss_node].gradient = 1.0;
    for (int node_index = loss_node; node_index >= 0; node_index--) {
        GraphNode *node = &graph_nodes[node_index];
        switch (node->operation) {
        case OPERATION_LEAF:
            break;
        case OPERATION_ADD:
            /* Addition passes the incoming gradient through unchanged. */
            graph_nodes[node->first_parent].gradient += node->gradient;
            graph_nodes[node->second_parent].gradient += node->gradient;
            break;
        case OPERATION_MULTIPLY:
            /* Each parent's slope is the OTHER parent's value. */
            graph_nodes[node->first_parent].gradient += graph_nodes[node->second_parent].data * node->gradient;
            graph_nodes[node->second_parent].gradient += graph_nodes[node->first_parent].data * node->gradient;
            break;
        case OPERATION_TANH:
            /* tanh's slope from its own output: 1 - tanh(z)^2. */
            graph_nodes[node->first_parent].gradient += (1.0 - node->data * node->data) * node->gradient;
            break;
        }
    }
}

/* ------------------------------------------------------------------ demos */

static double composed_function(double x) {
    return (2.0 * x + 1.0) * (2.0 * x + 1.0);
}

static void demonstrate_chain_rule(void) {
    const double small_step = 1e-3;
    double numerical_derivative =
        (composed_function(1.0 + small_step) - composed_function(1.0 - small_step)) / (2.0 * small_step);
    printf("1. Chain rule check: y = (2x+1)^2 at x=1\n");
    printf("   by the chain rule: dy/du * du/dx = 6 * 2 = 12\n");
    printf("   numerically:       %.3f\n", numerical_derivative);
}

static double loss_from_plain_numbers(double a, double b, double c) {
    return (a * b + c) * (a * b + c);
}

static void demonstrate_graph_backpropagation(void) {
    graph_node_count = 0;
    int input_a = create_leaf(2.0);
    int input_b = create_leaf(3.0);
    int input_c = create_leaf(-1.0);
    int product_u = multiply_values(input_a, input_b);
    int sum_v = add_values(product_u, input_c);
    int loss_l = multiply_values(sum_v, sum_v);
    run_backward_pass(loss_l);

    printf("\n2. Backpropagation through L = (a*b + c)^2 with a=2, b=3, c=-1\n");
    printf("   L = %.0f\n", graph_nodes[loss_l].data);
    printf("   dL/da = %.0f   dL/db = %.0f   dL/dc = %.0f   (the figure's numbers)\n",
           graph_nodes[input_a].gradient, graph_nodes[input_b].gradient, graph_nodes[input_c].gradient);

    /* The engine grades itself against Chapter 3's numerical method. */
    const double small_step = 1e-6;
    printf("   numerical re-check:");
    printf("  dL/da = %.3f",
           (loss_from_plain_numbers(2.0 + small_step, 3.0, -1.0) - loss_from_plain_numbers(2.0 - small_step, 3.0, -1.0)) / (2.0 * small_step));
    printf("  dL/db = %.3f",
           (loss_from_plain_numbers(2.0, 3.0 + small_step, -1.0) - loss_from_plain_numbers(2.0, 3.0 - small_step, -1.0)) / (2.0 * small_step));
    printf("  dL/dc = %.3f\n",
           (loss_from_plain_numbers(2.0, 3.0, -1.0 + small_step) - loss_from_plain_numbers(2.0, 3.0, -1.0 - small_step)) / (2.0 * small_step));
}

/* ------------------------------------------------------------ XOR training */

/* Real networks start their weights at small RANDOM values, never chosen by
 * hand: randomness breaks the symmetry between neurons so they can specialize
 * (Chapter 11 covers weight initialization). So we draw the 13 starting weights
 * randomly here too. This tiny linear congruential generator (LCG) is the exact
 * same one the Python version uses - same seed, same 64-bit arithmetic, same
 * sequence - so both languages start from identical weights. */
#define WEIGHT_INITIALIZATION_SEED 42u
#define INITIAL_WEIGHT_RANGE 1.0

static uint64_t random_generator_state;

static double next_uniform_weight(double low, double high) {
    /* Advance the 64-bit state (unsigned overflow wraps mod 2^64, matching
     * Python's mask), then map its top 53 bits to [0, 1) and rescale. */
    random_generator_state = random_generator_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double unit_interval = (double)(random_generator_state >> 11) * (1.0 / 9007199254740992.0);
    return low + unit_interval * (high - low);
}

static const double xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double xor_targets[4] = {-1.0, 1.0, 1.0, -1.0};

#define PARAMETER_COUNT 13

static void train_xor_network(void) {
    /* The 13 parameters live at arena indices 0..12 for the whole training
     * run; each epoch the arena is reset to just past them, so the forward
     * graph is rebuilt fresh while the parameters persist. */
    graph_node_count = 0;
    int parameter_nodes[PARAMETER_COUNT];
    int parameter_index = 0;
    /* Draw the 13 weights randomly, in the same order as Python: 3 hidden
     * neurons (each [x1 weight, x2 weight, bias]), then the output neuron. */
    random_generator_state = WEIGHT_INITIALIZATION_SEED;
    for (int neuron_index = 0; neuron_index < 3; neuron_index++) {
        for (int component_index = 0; component_index < 3; component_index++) {
            parameter_nodes[parameter_index++] =
                create_leaf(next_uniform_weight(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE));
        }
    }
    for (int component_index = 0; component_index < 4; component_index++) {
        parameter_nodes[parameter_index++] =
            create_leaf(next_uniform_weight(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE));
    }

    const double learning_rate = 0.2;
    const int epochs_to_print[] = {0, 10, 50, 100, 500, 1000, 2000};
    printf("\n3. Training the 2-3-1 tanh network on XOR (targets -1 and +1)\n");
    printf("   epoch   loss       predictions for (0,0) (0,1) (1,0) (1,1)\n");

    for (int epoch_number = 0; epoch_number <= 2000; epoch_number++) {
        graph_node_count = PARAMETER_COUNT;
        for (int i = 0; i < PARAMETER_COUNT; i++) {
            graph_nodes[parameter_nodes[i]].gradient = 0.0;
        }

        double predictions_this_epoch[4];
        int total_loss = create_leaf(0.0);
        for (int example_index = 0; example_index < 4; example_index++) {
            int first_input = create_leaf(xor_inputs[example_index][0]);
            int second_input = create_leaf(xor_inputs[example_index][1]);

            /* Forward pass, mirroring the Python network_forward() exactly:
             * three hidden tanh neurons, then one tanh output neuron. */
            int hidden_activations[3];
            for (int neuron_index = 0; neuron_index < 3; neuron_index++) {
                int weight_x1 = parameter_nodes[neuron_index * 3 + 0];
                int weight_x2 = parameter_nodes[neuron_index * 3 + 1];
                int bias = parameter_nodes[neuron_index * 3 + 2];
                int weighted_sum = add_values(
                    add_values(multiply_values(weight_x1, first_input),
                               multiply_values(weight_x2, second_input)),
                    bias);
                hidden_activations[neuron_index] = tanh_value(weighted_sum);
            }
            int output_weighted_sum = add_values(
                add_values(
                    add_values(multiply_values(parameter_nodes[9], hidden_activations[0]),
                               multiply_values(parameter_nodes[10], hidden_activations[1])),
                    multiply_values(parameter_nodes[11], hidden_activations[2])),
                parameter_nodes[12]);
            int prediction = tanh_value(output_weighted_sum);
            predictions_this_epoch[example_index] = graph_nodes[prediction].data;

            /* error = prediction - target, built as prediction + target*(-1)
             * exactly like the Python __sub__ does. */
            int negated_target = create_leaf(-xor_targets[example_index]);
            int prediction_error = add_values(prediction, negated_target);
            total_loss = add_values(total_loss, multiply_values(prediction_error, prediction_error));
        }

        run_backward_pass(total_loss);
        for (int i = 0; i < PARAMETER_COUNT; i++) {
            graph_nodes[parameter_nodes[i]].data -= learning_rate * graph_nodes[parameter_nodes[i]].gradient;
        }

        for (int print_index = 0; print_index < 7; print_index++) {
            if (epochs_to_print[print_index] == epoch_number) {
                printf("   %5d   %.6f   %+.3f  %+.3f  %+.3f  %+.3f\n",
                       epoch_number, graph_nodes[total_loss].data,
                       predictions_this_epoch[0], predictions_this_epoch[1],
                       predictions_this_epoch[2], predictions_this_epoch[3]);
            }
        }
    }
    printf("   -> XOR learned: outputs near -1, +1, +1, -1. No hand-wiring this time.\n");
}

int main(void) {
    demonstrate_chain_rule();
    demonstrate_graph_backpropagation();
    train_xor_network();
    return 0;
}
