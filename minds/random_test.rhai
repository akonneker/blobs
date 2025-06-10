// Create random boolean
let decision = rand_bool();

if decision {
    // Create random number
    let random_value = rand();
    print(`Random number = ${random_value}`);
} else {
    print("Fixed number = 42");
}

// Create array
let a = [1, 2, 3, 4, 5];

// Shuffle it!
a.shuffle();

// Now the array is shuffled randomly!
print(a);

// Sample a random value from the array
print(a.sample());

// Or sample multiple values
print(a.sample(3));