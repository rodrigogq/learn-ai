// Chapter 8 - the SAME autograd engine as ../c/tiny_autograd.c, but in C++.
//
// This file exists only as an optional side-by-side comparison (the course
// itself stays in C). The C version had to fake objects with a shared "arena"
// of structs referred to by integer indices, because C has no operator
// overloading and no automatic memory management. Here, C++ gives us both:
//
//   * a real Value object owns its place in the graph (a std::shared_ptr keeps
//     each node alive exactly as long as something still refers to it, so there
//     is no manual arena and no dangling indices), and
//   * operator overloading lets us write  a * b + c  and have the graph build
//     itself - reading almost exactly like the Python version.
//
// The result is byte-for-byte identical numbers to the C and Python versions
// (same seeded random weights), with code that is much closer to the Python.
//
// Build and run:
//     make -C chapters/08-backpropagation/cpp && ./chapters/08-backpropagation/cpp/build/tiny_autograd

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

// ------------------------------------------------------------------ the engine

// One node of the computation graph: its forward value, its gradient, the
// parents it was built from, and a small function holding the LOCAL derivative
// rule of the operation that made it. Exactly the four things the Python
// TrackedValue stored.
struct Node {
    double data;
    double gradient = 0.0;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> propagate_to_parents = [] {};
    explicit Node(double value) : data(value) {}
};

// Value is a thin handle around a shared_ptr<Node>. Copying a Value shares the
// same underlying node (that is what makes "a value used in two places" work),
// and when the last Value referring to a node goes out of scope, the node is
// freed automatically - no arena, no indices, no manual cleanup.
class Value {
public:
    std::shared_ptr<Node> node;

    // Non-explicit on purpose so a plain number becomes a leaf Value wherever
    // one is expected, letting us write  weight * x  with x a double.
    Value(double value) : node(std::make_shared<Node>(value)) {}
    explicit Value(std::shared_ptr<Node> existing) : node(std::move(existing)) {}

    double data() const { return node->data; }
    double gradient() const { return node->gradient; }
};

// Each operation computes its forward value AND records how to send gradient
// back to its parents - the closure captured below is this node's local rule.
// (We capture parent/result by raw pointer; the shared_ptrs in `parents` keep
// them alive, and capturing the result by value would create an ownership cycle.)

Value operator+(const Value &left, const Value &right) {
    auto result = std::make_shared<Node>(left.node->data + right.node->data);
    result->parents = {left.node, right.node};
    Node *l = left.node.get();
    Node *r = right.node.get();
    Node *out = result.get();
    // Addition passes the incoming gradient through to both parents unchanged.
    result->propagate_to_parents = [l, r, out] {
        l->gradient += out->gradient;
        r->gradient += out->gradient;
    };
    return Value(result);
}

Value operator*(const Value &left, const Value &right) {
    auto result = std::make_shared<Node>(left.node->data * right.node->data);
    result->parents = {left.node, right.node};
    Node *l = left.node.get();
    Node *r = right.node.get();
    Node *out = result.get();
    // Multiplication's local rule: each input's slope is the OTHER input.
    result->propagate_to_parents = [l, r, out] {
        l->gradient += r->data * out->gradient;
        r->gradient += l->data * out->gradient;
    };
    return Value(result);
}

Value operator-(const Value &left, const Value &right) {
    // Built from the operations above (a - b = a + b * -1), so it needs no
    // backward rule of its own - the graph handles it.
    return left + right * Value(-1.0);
}

Value tanh_value(const Value &input) {
    double t = std::tanh(input.node->data);
    auto result = std::make_shared<Node>(t);
    result->parents = {input.node};
    Node *in = input.node.get();
    Node *out = result.get();
    // tanh's slope is expressible from its own OUTPUT: 1 - tanh(z)^2.
    result->propagate_to_parents = [in, out, t] {
        in->gradient += (1.0 - t * t) * out->gradient;
    };
    return Value(result);
}

// Backpropagation: list the nodes so every parent comes before its children
// (a topological sort), seed dL/dL = 1, then apply each local rule in reverse.
void run_backward_pass(const Value &loss) {
    std::vector<std::shared_ptr<Node>> nodes_in_construction_order;
    std::unordered_set<Node *> already_visited;
    std::function<void(const std::shared_ptr<Node> &)> visit_parents_first =
        [&](const std::shared_ptr<Node> &value) {
            if (already_visited.count(value.get())) return;
            already_visited.insert(value.get());
            for (const auto &parent : value->parents) visit_parents_first(parent);
            nodes_in_construction_order.push_back(value);
        };
    visit_parents_first(loss.node);

    loss.node->gradient = 1.0;
    for (auto it = nodes_in_construction_order.rbegin(); it != nodes_in_construction_order.rend(); ++it) {
        (*it)->propagate_to_parents();
    }
}

// ----------------------------------------------- reproducible random weights

// The same tiny linear congruential generator (LCG) the C and Python versions
// use - same seed, same 64-bit arithmetic - so all three start from identical
// weights and print identical numbers.
class ReproducibleRandomGenerator {
public:
    explicit ReproducibleRandomGenerator(uint64_t seed) : state_(seed) {}
    double next_uniform(double low, double high) {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        double unit_interval = static_cast<double>(state_ >> 11) * (1.0 / 9007199254740992.0);
        return low + unit_interval * (high - low);
    }

private:
    uint64_t state_;
};

// ------------------------------------------------------------- demonstrations

void demonstrate_graph_backpropagation() {
    // Backpropagate L = (a*b + c)^2 with a=2, b=3, c=-1 - the figure's example.
    Value a = 2.0, b = 3.0, c = -1.0;
    Value u = a * b;      // = 6
    Value v = u + c;      // = 5
    Value loss = v * v;   // = 25
    run_backward_pass(loss);

    printf("1. Backpropagation through L = (a*b + c)^2 with a=2, b=3, c=-1\n");
    printf("   L = %.0f\n", loss.data());
    printf("   dL/da = %.0f   dL/db = %.0f   dL/dc = %.0f   (the figure's numbers)\n",
           a.gradient(), b.gradient(), c.gradient());
}

void train_xor_network() {
    const double LEARNING_RATE = 0.2;
    const double INITIAL_WEIGHT_RANGE = 1.0;
    const uint64_t WEIGHT_INITIALIZATION_SEED = 42;

    const double xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double xor_targets[4] = {-1.0, 1.0, 1.0, -1.0};

    // Draw the 13 weights randomly, in the same order as Python/C: 3 hidden
    // neurons (each [x1 weight, x2 weight, bias]), then the output neuron.
    ReproducibleRandomGenerator weight_generator(WEIGHT_INITIALIZATION_SEED);
    auto random_weight = [&] { return Value(weight_generator.next_uniform(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE)); };

    std::vector<std::vector<Value>> hidden_neuron_parameters;
    for (int neuron = 0; neuron < 3; neuron++) {
        hidden_neuron_parameters.push_back({random_weight(), random_weight(), random_weight()});
    }
    std::vector<Value> output_neuron_parameters = {random_weight(), random_weight(), random_weight(), random_weight()};

    // Gather every parameter so we can zero and update them each epoch.
    std::vector<Value> all_parameters;
    for (auto &neuron : hidden_neuron_parameters)
        for (auto &parameter : neuron) all_parameters.push_back(parameter);
    for (auto &parameter : output_neuron_parameters) all_parameters.push_back(parameter);

    auto network_forward = [&](double first_input, double second_input) {
        std::vector<Value> hidden_activations;
        for (auto &neuron : hidden_neuron_parameters) {
            Value weighted_sum = neuron[0] * first_input + neuron[1] * second_input + neuron[2];
            hidden_activations.push_back(tanh_value(weighted_sum));
        }
        Value output_weighted_sum = output_neuron_parameters[0] * hidden_activations[0] +
                                    output_neuron_parameters[1] * hidden_activations[1] +
                                    output_neuron_parameters[2] * hidden_activations[2] +
                                    output_neuron_parameters[3];
        return tanh_value(output_weighted_sum);
    };

    printf("\n2. Training the 2-3-1 tanh network on XOR (targets -1 and +1)\n");
    printf("   epoch   loss       predictions for (0,0) (0,1) (1,0) (1,1)\n");
    const std::unordered_set<int> epochs_to_print = {0, 10, 50, 100, 500, 1000, 2000};
    for (int epoch_number = 0; epoch_number <= 2000; epoch_number++) {
        double predictions_this_epoch[4];
        Value total_loss = 0.0;
        for (int example_index = 0; example_index < 4; example_index++) {
            Value prediction = network_forward(xor_inputs[example_index][0], xor_inputs[example_index][1]);
            predictions_this_epoch[example_index] = prediction.data();
            Value error = prediction - Value(xor_targets[example_index]);
            total_loss = total_loss + error * error;
        }

        for (auto &parameter : all_parameters) parameter.node->gradient = 0.0;
        run_backward_pass(total_loss);
        for (auto &parameter : all_parameters) parameter.node->data -= LEARNING_RATE * parameter.node->gradient;

        if (epochs_to_print.count(epoch_number)) {
            printf("   %5d   %.6f   %+.3f  %+.3f  %+.3f  %+.3f\n", epoch_number, total_loss.data(),
                   predictions_this_epoch[0], predictions_this_epoch[1],
                   predictions_this_epoch[2], predictions_this_epoch[3]);
        }
    }
    printf("   -> XOR learned: outputs near -1, +1, +1, -1. Same numbers as the C version,\n");
    printf("      but the graph built itself from  a * b + c  with no arena and no indices.\n");
}

int main() {
    demonstrate_graph_backpropagation();
    train_xor_network();
    return 0;
}
