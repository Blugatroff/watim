" Mostly stolen from https://gitlab.com/tsoding/porth/-/blob/master/editor/porth.vim
"
" Vim syntax file
" Language: Watim

" Usage Instructions
" Put this file in .vim/syntax/watim.vim
" and add in your .vimrc file the next line:
" autocmd BufRead,BufNewFile *.porth set filetype=porth

if exists("b:current_syntax")
  finish
endif

"set iskeyword=a-z,A-Z,-,*,_,!,@
syntax keyword watimTodos TODO

" Language keywords
syntax keyword watimKeywords struct if extern loop local fn memory data break else import as

" Comments
syntax region watimCommentLine start="//" end="$" contains=watimTodos

" String literals
syntax region watimString start=/\v"/ skip=/\v\\./ end=/\v"/ contains=watimEscapes

" Escape literals \n, \r, ....
syntax match watimEscapes display contained "\\[nr\"']"

" Number literals
syn match watimNumber "\<\d\+\>"

syntax match watimVariable "\$[0-9,a-z,A-Z,\-,*,_,!,@]*"
syntax match watimAssign "\#[0-9,a-z,A-Z,\-,*,_,!,@]*"

" Type names the compiler recognizes
syntax keyword watimTypeNames i64 i32 bool

" Set highlights
highlight default link watimTodos Todo
highlight default link watimKeywords Keyword
highlight default link watimCommentLine Comment
highlight default link watimString String
highlight default link watimNumber Constant
highlight default link watimTypeNames Type
highlight default link watimChar Character
highlight default link watimEscapes SpecialChar
highlight default link watimVariable Identifier
highlight default link watimAssign Identifier
highlight default link watimWord Function

let b:current_syntax = "watim"
