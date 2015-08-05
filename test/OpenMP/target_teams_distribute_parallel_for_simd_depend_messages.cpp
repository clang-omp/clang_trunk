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
  int b[1024];
  typedef float V __attribute__((vector_size(16)));
  V a;

  #pragma omp target teams distribute parallel for simd depend // expected-error {{expected '(' after 'depend'}} expected-error {{expected dependence type 'in', 'out' or 'inout'}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend ( // expected-error {{expected dependence type 'in', 'out' or 'inout'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend () // expected-error {{expected dependence type 'in', 'out' or 'inout'}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (argc // expected-error {{expected dependence type 'in', 'out' or 'inout'}} expected-error {{expected ':' in 'depend' clause}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : argc)) // expected-warning {{extra tokens at the end of '#pragma omp target teams distribute parallel for simd' are ignored}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (out: ) // expected-error {{expected expression}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (inout : foobool(argc)), depend (in, argc) // expected-error {{argument expression must be an l-value}} expected-error {{expected ':' in 'depend' clause}} expected-error {{expected expression}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (out :S1) // expected-error {{'S1' does not refer to a value}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : argv[1][1]='2') // expected-error {{expected variable name or an array item}} 
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : vec[1:2]) // expected-error {{argument expression must be an l-value}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : argv[0:-1]) // expected-error {{length of the array section must be greater than 0}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : argv[:]) // expected-error {{cannot define default length for non-array type 'char **'}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend (in : argv[3:4:1]) // expected-error {{expected ']'}} expected-note {{to match this '['}}
  for (int i = 0; i < 1024; i++) b[i]=1;
  #pragma omp target teams distribute parallel for simd depend(in:a[0:1]) // expected-error{{extended array notation is not allowed}}
  for (int i = 0; i < 1024; i++) b[i]=1;

  foo();
  return 0;
}
