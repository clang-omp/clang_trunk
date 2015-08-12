// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -o - %s

#pragma omp // expected-error {{expected an OpenMP directive}}
#pragma omp unknown_directive // expected-error {{expected an OpenMP directive}}

void foo() {
#pragma omp // expected-error {{expected an OpenMP directive}}
#pragma omp unknown_directive // expected-error {{expected an OpenMP directive}}
#pragma omp parallel unknown_clause // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
#pragma omp parallel ordered // expected-error {{unexpected OpenMP clause 'ordered' in directive '#pragma omp parallel'}}
#pragma omp for unknown_clause // expected-warning {{extra tokens at the end of '#pragma omp for' are ignored}}
for (int i = 0; i < 1; ++i) ++i;
#pragma omp for default(none) // expected-error {{unexpected OpenMP clause 'default' in directive '#pragma omp for'}}
for (int i = 0; i < 1; ++i) ++i;
foo();
}

typedef struct S {
#pragma omp parallel for private(j) schedule(static) if (tree1->totleaf > 1024) // expected-error {{unexpected OpenMP directive '#pragma omp parallel for'}}
} St;

