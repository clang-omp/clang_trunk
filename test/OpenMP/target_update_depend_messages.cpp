// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

class vector {
  public:
    int operator[](int index) { return 0; }
};

int main(int argc, char **argv) {
  vector vec;
  typedef float V __attribute__((vector_size(16)));
  V a;

  #pragma omp target update depend // expected-error {{expected '(' after 'depend'}} expected-error {{expected dependence type 'in', 'out' or 'inout'}}
  #pragma omp target update depend ( // expected-error {{expected dependence type 'in', 'out' or 'inout'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update depend () // expected-error {{expected dependence type 'in', 'out' or 'inout'}}
  #pragma omp target update depend (argc // expected-error {{expected dependence type 'in', 'out' or 'inout'}} expected-error {{expected ':' in 'depend' clause}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  #pragma omp target update depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target update' are ignored}}
  #pragma omp target update depend (out: ) // expected-error {{expected expression}}
  #pragma omp target update depend (inout : foobool(argc)), depend (in, argc) // expected-error {{argument expression must be an l-value}} expected-error {{expected ':' in 'depend' clause}} expected-error {{expected expression}}
  #pragma omp target update depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  #pragma omp target update depend (in : argv[1][1]='2') // expected-error {{expected variable name or an array item}} 
  #pragma omp target update depend (in : vec[1:2]) // expected-error {{argument expression must be an l-value}}
  #pragma omp target update depend (in : argv[0:-1]) // expected-error {{length of the array section must be greater than 0}}
  #pragma omp target update depend (in : argv[:]) // expected-error {{cannot define default length for non-array type 'char **'}}
  #pragma omp target update depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  #pragma omp target update depend(in:a[0:1]) // expected-error{{extended array notation is not allowed}}
  foo();

  return 0;
}
