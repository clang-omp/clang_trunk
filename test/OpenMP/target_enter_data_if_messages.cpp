// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

struct S1; // expected-note {{declared here}}

int main(int argc, char **argv) {
  #pragma omp target enter data if // expected-error {{expected '(' after 'if'}} expected-error {{expected expression}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if () // expected-error {{expected expression}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (argc)) // expected-warning {{extra tokens at the end of '#pragma omp target enter data' are ignored}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (argc > 0 ? argv[1] : argv[2]) // expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (foobool(argc)), if (true) // expected-error {{directive '#pragma omp target enter data' cannot contain more than one 'if' clause}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (S1) // expected-error {{'S1' does not refer to a value}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data if (argv[1]=2) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one map clause}}
  foo();

  return 0;
}
