// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s

void foo() {
}

bool foobool(int argc) {
  return argc;
}

int main(int argc, char **argv) {
  const int d = 5;
  const int da[5] = { 0 };
  int i;
  int &j = i;
  int *k = &j;
  const int (&l)[5] = da;
  #pragma omp target enter data map // expected-error {{expected '(' after 'map'}} expected-error {{expected expression}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map ( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map () // expected-error {{expected expression}} expected-error {{expected at least one map clause}}
  foo();
  #pragma omp target enter data map (release: argc) // expected-error {{invalid map-type}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map (delete: argc) // expected-error {{invalid map-type}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map (alloc) // expected-error {{expected expression}}  expected-error {{expected ':' in 'map' clause}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map (to argc // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error{{expected ':' in 'map' clause}}
  #pragma omp target enter data map (from: argc, // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{invalid map-type}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map (tofrom: argc > 0 ? argv[1] : argv[2]) // expected-error {{expected variable name}} expected-error {{expected at least one map clause}}
  #pragma omp target enter data map (argc) // expected-error {{invalid map-type}} expected-error {{expected at least one map clause}}
  foo();

  return 0;
}
