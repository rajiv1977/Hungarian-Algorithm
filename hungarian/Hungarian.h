#pragma once

#include <algorithm>
#include <deque>
#include <Eigen/Dense>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

typedef float                          float32_t;
typedef double                         float64_t;
typedef std::numeric_limits<float32_t> info;

template <typename T>
class Hungarian
{
  protected:
    struct Assignment_t
    {
        Eigen::Vector<uint32_t, Eigen::Dynamic> assignment;
        T                                       cost;
    };
    struct Outerplus_t
    {
        T                                       minval;
        Eigen::Vector<uint32_t, Eigen::Dynamic> rIdx;
        Eigen::Vector<uint32_t, Eigen::Dynamic> cIdx;
    };

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mCostMatrix;
    Assignment_t                                     mOverAllCostAndDataAssignments;
    bool                                             mStatus;

  public:
    Hungarian() = default;
    explicit Hungarian(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costMatrix)
        : mCostMatrix(costMatrix)
        , mStatus(false)
        , mOverAllCostAndDataAssignments{}
    {
        mStatus = (static_cast<uint32_t>(mCostMatrix.rows()) > 0) && (static_cast<uint32_t>(mCostMatrix.cols()) > 0);
        if (mStatus)
        {
            mOverAllCostAndDataAssignments = process(mCostMatrix);
        }
    }

    ~Hungarian() = default;

    Eigen::Vector<uint32_t, Eigen::Dynamic> getAssignmentIDs() const
    {
        return mOverAllCostAndDataAssignments.assignment;
    }

    T getOverAllCost() const
    {
        return mOverAllCostAndDataAssignments.cost;
    }

    bool getStatus() const
    {
        return mStatus;
    }

    Assignment_t process(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& costMat)
    {
        Eigen::Vector<uint32_t, Eigen::Dynamic> assignment =
            Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(costMat.rows());

        T cost = info::infinity();

        for (uint32_t ii = 0; ii < costMat.rows(); ii++)
        {
            auto row = costMat.row(ii);
            std::transform(row.begin(),
                           row.end(),
                           row.begin(),
                           row.begin(),
                           [](auto r1, auto r2)
                           {
                               if (r1 != r2)
                               {
                                   return info::infinity();
                               }
                               else
                               {
                                   return r1;
                               }
                           });
            costMat.block(ii, 0, 1, costMat.cols()) = row;
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> validMat = costMat;
        for (uint32_t ii = 0; ii < validMat.rows(); ii++)
        {
            auto row = validMat.row(ii);
            std::transform(row.begin(),
                           row.end(),
                           row.begin(),
                           [](auto r)
                           {
                               if (r < info::infinity())
                               {
                                   return 1;
                               }
                               else
                               {
                                   return 0;
                               }
                           });
        }

        Eigen::Vector<uint32_t, Eigen::Dynamic> validCol =
            Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(validMat.cols());
        for (uint32_t ii = 0; ii < validMat.cols(); ii++)
        {
            auto col     = validMat.col(ii);
            validCol(ii) = (std::any_of(col.begin(), col.end(), [](auto c) { return c == 1; }) == true) ? 1 : 0;
        }

        Eigen::Vector<uint32_t, Eigen::Dynamic> validRow =
            Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(validMat.rows());
        for (uint32_t ii = 0; ii < validMat.rows(); ii++)
        {
            auto row     = validMat.row(ii);
            validRow(ii) = (std::any_of(row.begin(), row.end(), [](auto r) { return r == 1; }) == true) ? 1 : 0;
        }

        auto nRows = std::accumulate(validRow.begin(), validRow.end(), 0);

        auto nCols = std::accumulate(validCol.begin(), validCol.end(), 0);

        auto n = std::max(nRows, nCols);
        if (n > 0)
        {
            Eigen::Vector<T, Eigen::Dynamic> maxValidation    = Eigen::Vector<T, Eigen::Dynamic>::Zero(costMat.cols());
            std::vector<T>                   maxValidationAll = {};
            for (uint32_t ii = 0; ii < costMat.cols(); ii++)
            {
                auto col1 = validMat.col(ii);
                auto col2 = costMat.col(ii);

                std::transform(col1.begin(),
                               col1.end(),
                               col2.begin(),
                               maxValidation.begin(),
                               [](auto c1, auto c2)
                               {
                                   if (c1 == 1)
                                   {
                                       return c2;
                                   }
                                   else
                                   {
                                       return -info::infinity();
                                   }
                               });
                if (maxValidation.rows() > 0)
                {
                    for (const auto& m : maxValidation)
                    {
                        if (m > -info::infinity())
                        {
                            maxValidationAll.push_back(m);
                        }
                    }
                }
            }
            auto minMax = std::minmax_element(maxValidationAll.begin(), maxValidationAll.end());
            auto maxv   = 10 * *minMax.second;

            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dMat =
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones(n, n) * maxv;

            uint32_t outter  = 0;
            uint32_t outterd = 0;
            for (const auto& v : validRow)
            {
                uint32_t inner  = 0;
                uint32_t innerd = 0;
                for (const auto& c : validCol)
                {
                    if (v == 1 && c == 1)
                    {
                        dMat(outterd, innerd) = costMat(outter, inner);
                        innerd++;
                    }
                    inner++;
                }
                if (v == 1)
                {
                    outterd++;
                }
                outter++;
            }

            //******************************************************
            // STEP 1 : Subtract the row minimum from each row.
            //******************************************************
            auto minR = computeMinimumCostOfRows(dMat, 1);
            auto minC = computeMinimumCostOfRows(bsxfun(dMat, minR, "minus"), 2);

            //**************************************************************************
            //  STEP 2: Find a zero of dMat. If there are no starred zeros in its
            //          column or row start the zero. Repeat for each zero
            //**************************************************************************
            auto zP = isEqualMatrix(bsxfun(minC, minR, "plus"), dMat);

            Eigen::Vector<uint32_t, Eigen::Dynamic> starZ = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(n);

            uint32_t trackColIndex = 0;
            auto     zP_v          = columnsOfMatrix(zP);
            while (std::any_of(zP_v.begin(), zP_v.end(), [](auto v) { return v == 1; }))
            {
                auto     colVec      = zP.col(trackColIndex);
                auto     zP_unity    = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(zP.rows(), zP.cols());
                uint32_t r           = 0;
                uint32_t c           = 0;
                auto     firstPosOne = std::find(colVec.begin(), colVec.end(), 1);
                if (firstPosOne != colVec.end())
                {
                    r = firstPosOne - colVec.begin() + 1;
                    c = trackColIndex + 1;
                    starZ(r - 1) = c;
                    zP.block(0, c - 1, zP.rows(), 1) = zP_unity.block(0, c - 1, zP.rows(), 1);
                    zP.block(r - 1, 0, 1, zP.cols()) = zP_unity.block(r - 1, 0, 1, zP.cols());
                }
                zP_v = columnsOfMatrix(zP);
                trackColIndex++;
            }

            uint32_t                                step = 0;
            Eigen::Vector<uint32_t, Eigen::Dynamic> uZr;
            Eigen::Vector<uint32_t, Eigen::Dynamic> uZc;
            uZr.resize(0);
            uZc.resize(0);

            while (1)
            {
                //**************************************************************************
                //  STEP 3 : Cover each column with a starred zero.If all the columns are
                //           covered then the matching is maximum
                //**************************************************************************
                if (std::all_of(starZ.begin(), starZ.end(), [](auto s) { return s > 0; }))
                {
                    break;
                }

                Eigen::Vector<uint32_t, Eigen::Dynamic> coverColumn = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(n);
                std::vector<uint32_t> regT = {};
                for (auto& s : starZ)
                {
                    if (s > 0)
                        regT.push_back(s);
                }
                if (regT.size() > 0)
                {
                    for (auto& r : regT)
                    {
                        coverColumn(r - 1) = true;
                    }
                }

                Eigen::Vector<uint32_t, Eigen::Dynamic> coverRow = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(n);

                Eigen::Vector<uint32_t, Eigen::Dynamic> primeZ = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(n);

                auto dMatNeg =
                    extract(dMat, custom<uint32_t>(coverRow, "negate"), custom<uint32_t>(coverColumn, "negate"));
                auto bsxNeg = bsxfun(extract(minC, custom<uint32_t>(coverColumn, "negate")),
                                     extract(minR, custom<uint32_t>(coverRow, "negate")),
                                     "plus");
                Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bsxFind =
                    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(dMatNeg.rows(), dMatNeg.cols());
                for (uint32_t jj = 0; jj < dMatNeg.cols(); jj++)
                {
                    for (uint32_t ii = 0; ii < dMatNeg.rows(); ii++)
                    {
                        if (dMatNeg(ii, jj) == bsxNeg(ii, 0))
                        {
                            bsxFind(ii, jj) = 1;
                        }
                    }
                }
                std::vector<uint32_t> trIdx = {};
                for (uint32_t jj = 0; jj < bsxFind.cols(); jj++)
                {
                    for (uint32_t ii = 0; ii < bsxFind.rows(); ii++)
                    {
                        if (bsxFind(ii, jj) == 1)
                            trIdx.push_back(ii + 1);
                    }
                }
                std::vector<uint32_t> tcIdx = {};
                for (uint32_t ii = 0; ii < bsxFind.rows(); ii++)
                {
                    for (uint32_t jj = 0; jj < bsxFind.cols(); jj++)
                    {
                        if (bsxFind(ii, jj) == 1)
                            tcIdx.push_back(jj + 1);
                    }
                }
                Eigen::Vector<uint32_t, Eigen::Dynamic> rIdx;
                rIdx.resize(0);
                if (trIdx.size() > 0)
                {
                    rIdx.resize(trIdx.size());
                    rIdx = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(trIdx.size());
                    std::copy(trIdx.begin(), trIdx.end(), rIdx.begin());
                }
                Eigen::Vector<uint32_t, Eigen::Dynamic> cIdx;
                cIdx.resize(0);
                if (tcIdx.size() > 0)
                {
                    cIdx.resize(tcIdx.size());
                    cIdx = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(tcIdx.size());
                    std::copy(tcIdx.begin(), tcIdx.end(), cIdx.begin());
                }

                while (1)
                {
                    //**************************************************************************
                    //   STEP 4 :  Find a noncovered zero and prime it.If there is no starred
                    //             zero in the row containing this primed zero, Go to Step 5.
                    //             Otherwise, cover this row and uncover the column containing
                    //             the starred zero.Continue in this manner until there are no
                    //             uncovered zeros left. Save the smallest uncovered value and
                    //             Go to Step 6. %
                    //**************************************************************************
                    auto cR = findEQ<uint32_t>(custom<uint32_t>(coverRow, "negate"), 1);
                    auto cC = findEQ<uint32_t>(custom<uint32_t>(coverColumn, "negate"), 1);
                    rIdx = extractElementRespectIndexVector<uint32_t>(rIdx, cR);
                    cIdx = extractElementRespectIndexVector<uint32_t>(cIdx, cC);

                    step = 6;
                    while (cIdx.rows() > 0)
                    {
                        
                        uZr.resize(1);
                        uZr(0) = rIdx(0);
                        uZc.resize(1);
                        uZc(0) = cIdx(0);
                        std::deque<uint32_t> uZrc = {};
                        if (uZr.rows() > 0)
                        {
                            if (uZc.rows() > 0)
                            {
                                uint32_t ii = 0;
                                for (auto& u : uZr)
                                {
                                    primeZ(u - 1) = uZc(ii++);
                                }
                            }
                        }
                        auto stz = starZ(uZr(0) - 1);
                        if (!stz)
                        {
                            step = 5;
                            break;
                        }
                        
                        coverRow(uZr(0) - 1) = true;
                        coverColumn(stz - 1) = false;
                        Eigen::Vector<uint32_t, Eigen::Dynamic> z;
                        z.resize(0);
                        std::vector<uint32_t> tz = {};
                        for (uint32_t ii = 0; ii < rIdx.size(); ii++)
                        {
                            if (rIdx(ii) == uZr(0))
                            {
                                tz.push_back(1);
                            }
                            else
                            {
                                tz.push_back(0);
                            }
                        }
                        if (tz.size() > 0)
                        {
                            z.resize(tz.size());
                            std::copy(tz.begin(), tz.end(), z.begin());
                        }

                        rIdx = eliminateElementsRespectIndexVector<uint32_t>(rIdx, z);
                        cIdx = eliminateElementsRespectIndexVector<uint32_t>(cIdx, z);
                        cR = findEQ<uint32_t>(custom<uint32_t>(coverRow, "negate"), 1);
                        auto dstz   = extract(dMat, custom<uint32_t>(coverRow, "negate"), stz);
                        auto mCover = extract(minR, custom<uint32_t>(coverRow, "negate"));
                        auto mst    = minC(stz - 1);
                        std::transform(mCover.begin(), mCover.end(), mCover.begin(), [=](auto m) { return m + mst; });
                        std::vector<uint32_t> tempz = {};
                        for (uint32_t kk = 0; kk < dstz.rows(); kk++)
                        {
                            if (dstz(kk) == mCover(kk))
                            {
                                tempz.push_back(1);
                            }
                            else
                            {
                                tempz.push_back(0);
                            }
                        }
                        if (tempz.size() > 0)
                        {
                            z.resize(tempz.size());
                            z = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(tempz.size());
                            std::copy(tempz.begin(), tempz.end(), z.begin());
                        }
                        std::vector<uint32_t> rIdx_t = {};
                        rIdx_t.resize(rIdx.size());
                        std::copy(rIdx.begin(), rIdx.end(), rIdx_t.begin());
                        std::vector<uint32_t> cr_t    = {};
                        uint32_t              ezinter = 0;
                        for (auto& e : z)
                        {
                            if (e > 0)
                                cr_t.push_back(cR(ezinter));
                            ezinter++;
                        }

                        if (cr_t.size() > 0)
                        {
                            std::back_insert_iterator<std::vector<uint32_t>> back_it(rIdx_t);
                            std::copy(cr_t.begin(), cr_t.end(), back_it);
                            rIdx.resize(rIdx_t.size());
                            std::copy(rIdx_t.begin(), rIdx_t.end(), rIdx.begin());
                        }

                        std::vector<uint32_t> cIdx_t = {};
                        cIdx_t.resize(cIdx.size());
                        std::copy(cIdx.begin(), cIdx.end(), cIdx_t.begin());

                        auto sum = std::accumulate(z.begin(), z.end(), 0);
                        if (sum > 0)
                        {
                            Eigen::Vector<uint32_t, Eigen::Dynamic> stOnes =
                                Eigen::Vector<uint32_t, Eigen::Dynamic>::Ones(sum) * stz;
                            std::back_insert_iterator<std::vector<uint32_t>> back_it(cIdx_t);
                            std::copy(stOnes.begin(), stOnes.end(), back_it);

                            cIdx.resize(cIdx_t.size());
                            std::copy(cIdx_t.begin(), cIdx_t.end(), cIdx.begin());
                        }
                    }

                    if (step == 6)
                    {
                        //*************************************************************************************
                        //    STEP 6 :  Add the minimum uncovered value to every element of each covered
                        //              row, and subtract it from every element of each uncovered column.
                        //              Return to Step 4 without altering any stars, primes, or covered lines.
                        //*************************************************************************************
                        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eval1 = extract(
                            dMat, custom<uint32_t>(coverRow, "negate"), custom<uint32_t>(coverColumn, "negate"));
                        Eigen::Vector<T, Eigen::Dynamic> eval2 = extract(minR, custom<uint32_t>(coverRow, "negate"));
                        Eigen::Vector<T, Eigen::Dynamic> eval3 = extract(minC, custom<uint32_t>(coverColumn, "negate"));
                        auto [minval, rIdx1, cIdx1]            = outerplus(eval1, eval2, eval3);
                        if (rIdx1.rows() > 0)
                        {
                            rIdx.resize(rIdx1.rows());
                            rIdx = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(rIdx1.rows());
                            std::copy(rIdx1.begin(), rIdx1.end(), rIdx.begin());
                        }
                        if (cIdx1.size() > 0)
                        {
                            cIdx.resize(cIdx1.size());
                            cIdx = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(cIdx1.size());
                            std::copy(cIdx1.begin(), cIdx1.end(), cIdx.begin());
                        }
                        auto answer1 = custom<uint32_t>(coverColumn, "negate");
                        if (answer1.rows() > 0)
                        {
                            for (uint32_t ii = 0; ii < answer1.rows(); ii++)
                            {
                                if (answer1(ii) == 1)
                                {
                                    minC(ii) = minC(ii) + minval;
                                }
                            }
                        }
                        for (uint32_t ii = 0; ii < coverRow.rows(); ii++)
                        {
                            if (coverRow(ii) == 1)
                            {
                                minR(ii) = minR(ii) - minval;
                            }
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                //****************************************************************************************
                //    STEP 5 :  Construct a series of alternating primed and starred zeros
                //              as follows :
                //              Let Z0 represent the uncovered primed zero found in Step 4.
                //              Let Z1 denote the starred zero in the column of Z0(if any).
                //              Let Z2 denote the primed zero in the row of Z1(there will always
                //              be one). Continue until the series terminates at a primed zero
                //              that has no starred zero in its column.Unstar each starred
                //              zero of the series, star each primed zero of the series, erase
                //              all primes and uncover every line in the matrix.Return to Step 3.
                //****************************************************************************************
                auto rowZ1 = findEQ<uint32_t>(starZ, uZc(0));
                computeStarZ<uint32_t>(starZ, uZr, uZc);

                
                while (rowZIterater<uint32_t>(rowZ1) > 0)
                {
                    for (uint32_t ii = 0; ii < rowZ1.size(); ii++)
                    {
                        starZ(rowZ1(ii) - 1) = 0;
                    }
                    uZc = {};
                    uZc.resize(rowZ1.size());
                    uZc = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(rowZ1.size());
                    for (uint32_t ii = 0; ii < rowZ1.size(); ii++)
                    {
                        uZc(ii) = primeZ(rowZ1(ii) - 1);
                    }
                    uZr = {};
                    uZr.resize(rowZ1.size());
                    uZr = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(rowZ1.size());
                    uZr         = rowZ1;
                    auto z_inpt = uZc(0);
                    rowZ1 = findEQ<uint32_t>(starZ, z_inpt);
                    computeStarZ<uint32_t>(starZ, uZr, uZc);
                }
            }

            auto rowIdx = findNE<uint32_t>(validRow, 0);
            auto colIdx = findNE<uint32_t>(validCol, 0);
            Eigen::Vector<uint32_t, Eigen::Dynamic> copyStarZ = {};
            copyStarZ.resize(nRows);
            copyStarZ = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(nRows);
            for (uint32_t ii = 0; ii < static_cast<uint32_t>(nRows); ii++)
            {
                copyStarZ(ii) = starZ(ii);
            }
            starZ = {};
            starZ.resize(nRows);
            starZ = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(nRows);
            starZ = copyStarZ;
            auto vIdx = findSEQTraits<uint32_t>(starZ, nCols);
            computeAssignment<uint32_t>(assignment, rowIdx, colIdx, vIdx, starZ);
            cost = computeCost<uint32_t>(costMat, assignment);
        }
        return {assignment, cost};
    }

    Eigen::Vector<T, Eigen::Dynamic>
        computeMinimumCostOfRows(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost, uint32_t rc)
    {
        std::vector<T> vec = {};
        uint32_t       len = (rc == 1) ? cost.rows() : cost.cols();
        for (uint32_t ii = 0; ii < len; ii++)
        {
            Eigen::Vector<T, Eigen::Dynamic> ivec = {};
            if (rc == 1)
            {
                ivec = cost.row(ii);
            }
            else
            {
                ivec = cost.col(ii);
            }
            auto result = std::minmax_element(ivec.begin(), ivec.end());
            vec.push_back(*result.first);
        }
        Eigen::Vector<T, Eigen::Dynamic> out = {};
        out.resize(0);
        if (vec.size() > 0)
        {
            out.resize(vec.size());
            std::copy(vec.begin(), vec.end(), out.begin());
        }
        return out;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bsxfun(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M,
                                                            const Eigen::Vector<T, Eigen::Dynamic>&                 vec,
                                                            const std::string                                       str)
    {
        auto dM = M;
        if (str.compare("minus") == 0)
        {
            for (uint32_t ii = 0; ii < dM.cols(); ii++)
            {
                dM.block(0, ii, dM.rows(), 1) = dM.block(0, ii, dM.rows(), 1) - vec;
            }
        }
        if (str.compare("add") == 0)
        {
            for (uint32_t ii = 0; ii < dM.cols(); ii++)
            {
                dM.block(0, ii, dM.rows(), 1) = dM.block(0, ii, dM.rows(), 1) + vec;
            }
        }
        return dM;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> bsxfun(const Eigen::Vector<T, Eigen::Dynamic>& vec1,
                                                            const Eigen::Vector<T, Eigen::Dynamic>& vec2,
                                                            const std::string                       str)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(vec2.rows(), vec1.rows());
        if (str.compare("minus") == 0)
        {
            for (uint32_t ii = 0; ii < vec2.rows(); ii++)
            {
                for (uint32_t jj = 0; jj < vec1.rows(); jj++)
                {
                    M(ii, jj) = vec2(ii) - vec1(jj);
                }
            }
        }
        if (str.compare("plus") == 0)
        {
            for (uint32_t ii = 0; ii < vec2.rows(); ii++)
            {
                for (uint32_t jj = 0; jj < vec1.rows(); jj++)
                {
                    M(ii, jj) = vec2(ii) + vec1(jj);
                }
            }
        }
        return M;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
        isEqualMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M1,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M2)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> O =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(M1.rows(), M1.cols());

        for (uint32_t ii = 0; ii < M1.rows(); ii++)
        {
            for (uint32_t jj = 0; jj < M1.cols(); jj++)
            {
                if (M1(ii, jj) == M2(ii, jj))
                    O(ii, jj) = 1;
            }
        }
        return O;
    }

    std::vector<T> columnsOfMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M)
    {
        std::vector<T> v = {};
        for (uint32_t jj = 0; jj < static_cast<uint32_t>(M.cols()); jj++)
        {
            for (uint32_t ii = 0; ii < static_cast<uint32_t>(M.rows()); ii++)
            {
                v.push_back(M(ii, jj));
            }
        }
        return v;
    }

    template <typename U>
    Eigen::Vector<uint32_t, Eigen::Dynamic> custom(const Eigen::Vector<U, Eigen::Dynamic>& vec, const std::string str)
    {
        Eigen::Vector<uint32_t, Eigen::Dynamic> ivec = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(vec.rows());
        if (str.compare("negate") == 0)
        {
            for (uint32_t ii = 0; ii < vec.rows(); ii++)
            {
                if (vec(ii) == 0)
                    ivec(ii) = 1;
                else
                    ivec(ii) = 0;
            }
        }
        return ivec;
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> extract(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M,
                                                             const Eigen::Vector<uint32_t, Eigen::Dynamic>& vec1,
                                                             const Eigen::Vector<uint32_t, Eigen::Dynamic>& vec2)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> OO =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(M.rows(), M.cols());
        for (uint32_t ii = 0; ii < vec1.rows(); ii++)
        {
            for (uint32_t jj = 0; jj < vec2.rows(); jj++)
            {
                OO(ii, jj) = vec1(ii) * vec2(jj);
            }
        }

        uint32_t nRows = 0;
        for (uint32_t ii = 0; ii < OO.rows(); ii++)
        {
            auto row = OO.row(ii);
            if (std::any_of(row.begin(), row.end(), [](auto e) { return e == 1; }))
            {
                nRows++;
            }
        }

        uint32_t nCols = 0;
        for (uint32_t ii = 0; ii < OO.cols(); ii++)
        {
            auto col = OO.col(ii);
            if (std::any_of(col.begin(), col.end(), [](auto e) { return e == 1; }))
            {
                nCols++;
            }
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> fout =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(nRows, nCols);

        uint32_t outer  = 0;
        bool     status = false;
        for (uint32_t ii = 0; ii < M.rows(); ii++)
        {
            uint32_t inner = 0;
            for (uint32_t jj = 0; jj < M.cols(); jj++)
            {
                if (OO(ii, jj) == 1)
                {
                    status             = true;
                    fout(outer, inner) = M(ii, jj);
                    inner++;
                }
            }
            if (status)
            {
                status = false;
                outer++;
            }
        }
        return fout;
    }

    Eigen::Vector<T, Eigen::Dynamic> extract(const Eigen::Vector<T, Eigen::Dynamic>&        vec1,
                                             const Eigen::Vector<uint32_t, Eigen::Dynamic>& vec2)
    {
        Eigen::Vector<T, Eigen::Dynamic> fout;
        fout.resize(0);
        uint32_t numReg = 0;
        std::for_each(vec2.begin(),
                      vec2.end(),
                      [&](auto e)
                      {
                          if (e == 1)
                          {
                              numReg++;
                          }
                      });

        if (numReg > 0)
        {
            fout = Eigen::Vector<T, Eigen::Dynamic>::Zero(numReg);
        }

        uint32_t count = 0;
        for (uint32_t ii = 0; ii < vec2.rows(); ii++)
        {
            if (vec2(ii) == 1)
            {
                fout(count++) = vec1(ii);
            }
        }
        return fout;
    }

    Eigen::Vector<T, Eigen::Dynamic> extract(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& M,
                                             const Eigen::Vector<uint32_t, Eigen::Dynamic>&          vec1,
                                             const uint32_t&                                         scalar)
    {
        std::vector<T> vec   = {};
        auto           col   = M.col(scalar - 1);
        uint32_t       count = 0;
        for (const auto v : vec1)
        {
            if (v > 0)
            {
                vec.push_back(col(count));
            }
            count++;
        }

        Eigen::Vector<T, Eigen::Dynamic> fout;
        fout.resize(0);
        if (vec.size() > 0)
        {
            fout.resize(vec.size());
            std::copy(vec.begin(), vec.end(), fout.begin());
        }

        return fout;
    }

    template <typename W>
    Eigen::Vector<uint32_t, Eigen::Dynamic> findEQ(const Eigen::Vector<W, Eigen::Dynamic>& vec, const W& value)
    {
        Eigen::Vector<uint32_t, Eigen::Dynamic> out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(0);
        out.resize(0);
        std::vector<uint32_t> iv    = {};
        uint32_t              index = 0;
        for (const auto& v : vec)
        {
            if (v == value)
            {
                iv.push_back(index + 1);
            }
            index++;
        }
        if (iv.size() > 0)
        {
            out.resize(iv.size());
            out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(iv.size());
            copy(iv.begin(), iv.end(), out.begin());
        }
        return out;
    }

    template <typename U>
    Eigen::Vector<U, Eigen::Dynamic>
        extractElementRespectIndexVector(const Eigen::Vector<U, Eigen::Dynamic>& elementVec,
                                         const Eigen::Vector<U, Eigen::Dynamic>& indexVec)
    {
        std::vector<U> internalVec = {};
        for (U ii = 0; ii < elementVec.size(); ii++)
        {
            auto jj = elementVec(ii);
            internalVec.push_back(indexVec(jj - 1));
        }
        Eigen::Vector<U, Eigen::Dynamic> out = {};
        out.resize(internalVec.size());
        std::copy(internalVec.begin(), internalVec.end(), out.begin());

        return out;
    }

    template <typename U>
    Eigen::Vector<U, Eigen::Dynamic>
        eliminateElementsRespectIndexVector(const Eigen::Vector<U, Eigen::Dynamic>& elementVec,
                                            const Eigen::Vector<U, Eigen::Dynamic>& indexVec)
    {
        std::deque<U> deq = {};
        deq.resize(elementVec.size());
        std::copy(elementVec.begin(), elementVec.end(), deq.begin());
        for (const auto& i : indexVec)
        {
            if (i > 0)
            {
                deq.erase(deq.begin() + i - 1);
            }
        }
        Eigen::Vector<U, Eigen::Dynamic> out;
        out.resize(0);
        if (deq.size() > 0)
        {
            out.resize(deq.size());
            std::copy(deq.begin(), deq.end(), out.begin());
        }
        return out;
    }

    Outerplus_t outerplus(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& O,
                          const Eigen::Vector<T, Eigen::Dynamic>&                 x,
                          const Eigen::Vector<T, Eigen::Dynamic>&                 y)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> M = O;
        auto nx = M.rows();
        auto ny = M.cols();
        auto minval = info::infinity();
        Eigen::Vector<uint32_t, Eigen::Dynamic> rIdx;
        rIdx.resize(0);
        Eigen::Vector<uint32_t, Eigen::Dynamic> cIdx;
        cIdx.resize(0);

        for (uint32_t r = 0; r < nx; r++)
        {
            auto x1 = x(r);
            for (uint32_t c = 0; c < ny; c++)
            {
                M(r, c) = M(r, c) - (x1 + y(c));
                if (minval > M(r, c))
                {
                    minval = M(r, c);
                }
            }
        }
        std::vector<uint32_t> rIdx_t = {};
        for (uint32_t jj = 0; jj < M.cols(); jj++)
        {
            for (uint32_t ii = 0; ii < M.rows(); ii++)
            {
                if (M(ii, jj) == minval)
                    rIdx_t.push_back(ii + 1);
            }
        }
        std::vector<uint32_t> cIdx_t = {};
        for (uint32_t jj = 0; jj < M.cols(); jj++)
        {
            for (uint32_t ii = 0; ii < M.rows(); ii++)
            {
                if (M(ii, jj) == minval)
                    cIdx_t.push_back(jj + 1);
            }
        }
        if (rIdx_t.size() > 0)
        {
            rIdx.resize(rIdx_t.size());
            std::copy(rIdx_t.begin(), rIdx_t.end(), rIdx.begin());
        }
        if (cIdx_t.size() > 0)
        {
            cIdx.resize(cIdx_t.size());
            std::copy(cIdx_t.begin(), cIdx_t.end(), cIdx.begin());
        }
        return {minval, rIdx, cIdx};
    }

    template <typename U>
    Eigen::Vector<uint32_t, Eigen::Dynamic> findSEQTraits(const Eigen::Vector<U, Eigen::Dynamic>& vec, const U& value)
    {
        std::vector<uint32_t> iv;
        for (const auto& v : vec)
        {
            if (v <= value)
            {
                iv.push_back(1);
            }
            else
            {
                iv.push_back(0);
            }
        }

        Eigen::Vector<uint32_t, Eigen::Dynamic> out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(0);
        out.resize(0);
        if (iv.size() > 0)
        {
            out.resize(iv.size());
            out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(iv.size());
            copy(iv.begin(), iv.end(), out.begin());
        }
        return out;
    }

    template <typename U>
    void computeAssignment(Eigen::Vector<U, Eigen::Dynamic>&       assignment,
                           const Eigen::Vector<U, Eigen::Dynamic>& rowIdx,
                           const Eigen::Vector<U, Eigen::Dynamic>& colIdx,
                           const Eigen::Vector<U, Eigen::Dynamic>& vIdx,
                           const Eigen::Vector<U, Eigen::Dynamic>& starZ)
    {
        std::vector<U> iStarZ = {};
        U              id     = 0;
        for (const auto& v : vIdx)
        {
            if (v == 1)
            {
                iStarZ.push_back(starZ(id));
            }
            id++;
        }
        std::vector<U> iRowIdx = {};
        id                     = 0;
        for (const auto& v : vIdx)
        {
            if (v == 1)
            {
                iRowIdx.push_back(rowIdx(id));
            }
            id++;
        }
        for (U ii = 0; ii < findEQ<U>(vIdx, 1).rows(); ii++)
        {
            assignment(iRowIdx.at(ii) - 1) = colIdx(iStarZ.at(ii) - 1);
        }
    }

    template <typename W>
    Eigen::Vector<uint32_t, Eigen::Dynamic> findNE(const Eigen::Vector<W, Eigen::Dynamic>& vec, const W& value)
    {
        std::vector<W> holder;
        W              id    = 0.0F;
        auto           vec_t = vec.transpose();
        std::transform(vec_t.begin(),
                       vec_t.end(),
                       back_inserter(holder),
                       [&](const auto& v)
                       {
                           id++;
                           if (v != value)
                           {
                               return id;
                           }
                       });

        Eigen::Vector<uint32_t, Eigen::Dynamic> out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(0);
        out.resize(0);
        std::vector<uint32_t> iv;
        uint32_t              index = 0;
        for (const auto& v : vec)
        {
            if (v != value)
            {
                iv.push_back(index + 1);
            }
            index++;
        }

        if (iv.size() > 0)
        {
            out.resize(iv.size());
            out = Eigen::Vector<uint32_t, Eigen::Dynamic>::Zero(iv.size());
            copy(iv.begin(), iv.end(), out.begin());
        }
        return out;
    }

    template <typename U>
    T computeCost(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost,
                  const Eigen::Vector<U, Eigen::Dynamic>&                 assignment)
    {
        std::vector<U> assign = {};
        auto           ids    = findGT<U>(assignment, 0);
        for (const auto& id : ids)
        {
            assign.push_back(assignment(id - 1));
        }
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> iCost =
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(ids.size(), assign.size());
        U rows = 0;
        for (const auto& id : ids)
        {
            U cols = 0;
            for (const auto& a : assign)
            {

                iCost(rows, cols) = cost(id - 1, a - 1);
                cols++;
            }
            rows++;
        }
        return iCost.trace();
    }

    template <typename U>
    Eigen::Vector<int32_t, Eigen::Dynamic> findGT(const Eigen::Vector<U, Eigen::Dynamic>& vec, const U& value)
    {
        std::vector<int32_t> iv;
        U                    id = 0;
        for (const auto& v : vec)
        {
            if (v > value)
            {
                iv.push_back(id + 1);
            }
            id++;
        }
        Eigen::Vector<int32_t, Eigen::Dynamic> out = {};
        if (iv.size() > 0)
        {
            out.resize(iv.size());
            copy(iv.begin(), iv.end(), out.begin());
        }
        return out;
    }

    template <typename U>
    U rowZIterater(const Eigen::Vector<U, Eigen::Dynamic>& row)
    {
        U sum = 0;
        for (const auto& r : row)
        {
            if (r > 0)
            {
                sum++;
            }
        }
        return sum;
    }

    template <typename U>
    void computeStarZ_new(Eigen::Vector<U, Eigen::Dynamic>&       starZ,
                          const Eigen::Vector<U, Eigen::Dynamic>& uZr,
                          const Eigen::Vector<U, Eigen::Dynamic>& uZc)
    {
        if (uZr.size() == 1)
        {
            starZ(uZr(0) - 1) = uZc(0);
        }
        else
        {
            for (U ii = 0; ii < uZr.size(); ii++)
            {
                starZ(uZr(ii) - 1) = uZc(ii);
            }
        }
    }

    template <typename U>
    void computeStarZ(Eigen::Vector<U, Eigen::Dynamic>&       starZ,
                      const Eigen::Vector<U, Eigen::Dynamic>& uZr,
                      const Eigen::Vector<U, Eigen::Dynamic>& uZc)
    {
        if (uZr.size() == 1)
        {
            starZ(uZr(0) - 1) = uZc(0);
        }
        else
        {
            for (U ii = 0; ii < uZr.size(); ii++)
            {
                starZ(uZr(ii) - 1) = uZc(ii);
            }
        }
    }
};
