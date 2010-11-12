! pm.binomial_like(data_, survey_plan.n[i], invlogit(sp_sub))

      SUBROUTINE binomial(d, n, f, nf, lp)
cf2py intent(hide) nf
cf2py intent(out) lp
      DOUBLE PRECISION d,n,f(nf),lp(nf)
      INTEGER i

      do i=1,nf
          lp(i) = d*f(i)-n*dlog(1.0D0+dexp(f(i)))
      end do

      RETURN
      END